from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

DEFAULT_MAX_ATTEMPTS = 4
DEFAULT_MAX_USD_PER_HR = 1.0
VALID_CLOUDS = frozenset({"secure", "community"})

### PLACEMENT (cloud, datacenter) ###########################


def _entries_for_ids(
    catalog: Mapping[str, Mapping[str, Any]],
    gpu_ids: Sequence[str],
    cloud: str,
    *,
    require_price: bool,
) -> list[LadderEntry]:
    entries: list[LadderEntry] = []
    for gpu_id in gpu_ids:
        gpu = catalog.get(gpu_id)
        if gpu is None:
            raise ValueError(
                f"GPU {gpu_id!r} is not in the gpus catalog in runpod_config.yaml"
            )
        price = price_for(gpu, cloud)
        if price is None:
            if require_price:
                raise ValueError(
                    f"GPU {gpu_id!r} has no {cloud} rate in the catalog "
                    f"(not offered on that cloud)"
                )
            continue
        entries.append(
            LadderEntry(
                id=gpu_id,
                cloud=cloud,
                usd_per_hr=price,
                vram_gb=int(gpu["vram_gb"]),
            )
        )
    return entries


#############################################################

### CONFIG Parse (catalog index, retry caps, stage VRAM, YAML pin) 
    

def _retry_settings(
    cfg: Mapping[str, Any],
    max_usd_per_hr: float | None,
    max_attempts: int | None,
) -> tuple[float, int]:

    retry = cfg.get("retry") or {}
    if not isinstance(retry, Mapping):
        raise ValueError("retry config must be a mapping")

    cap = max_usd_per_hr if max_usd_per_hr is not None else retry.get("max_usd_per_hr", DEFAULT_MAX_USD_PER_HR)
    attempts = max_attempts if max_attempts is not None else retry.get("max_attempts", DEFAULT_MAX_ATTEMPTS)
    
    try:
        cap_f = float(cap)
    except ValueError:
        raise ValueError(f"invalid float caps: max_usd_per_hr={cap!r}")

    try:
        attempts_i = int(attempts)
    except ValueError:
        raise ValueError(f"invalid retry attempts: max_attempts={attempts!r}")

    if cap_f <= 0:
        raise ValueError(f"max_usd_per_hr must be > 0, got {cap_f}")

    if attempts_i < 1:
        raise ValueError(f"max_attempts must be >= 1, got {attempts_i}")
        
    return cap_f, attempts_i


def _catalog_by_id(cfg: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:

    by_id: dict[str, Mapping[str, Any]] = {}

    gpus = cfg.get("gpus")
    if not isinstance(gpus, list) or not gpus:
        raise ValueError("runpod_config.yaml must declare a non-empty gpus catalog")
    
    for entry in gpus:

        if not isinstance(entry, Mapping):
            raise ValueError(f"gpus[] entries must be mappings, got {entry!r}")

        gpu_id = entry.get("id")
        if not isinstance(gpu_id, str) or not gpu_id.strip():
            raise ValueError(f"gpus[].id must be a non-empty string, got {gpu_id!r}")

        if gpu_id in by_id:
            continue

        by_id[gpu_id] = entry

    return by_id


def _parse_vram_bound(
    value: Any,
    *,
    field: str,
    default: int | None,
    allow_none: bool,
) -> int | None:
    if value is None:
        if allow_none:
            return None
        value = default
    try:
        parsed = int(value)
    except ValueError:
        raise ValueError(f"{field} must be an int, got {value!r}")
    if parsed < 0:
        raise ValueError(f"{field} must be >= 0, got {parsed}")
    return parsed


def _stage_vram_bounds(
    cfg: Mapping[str, Any],
    stage: str,
    *,
    min_vram_gb: int | None = None,
    max_vram_gb: int | None = None,
) -> tuple[int, int | None]:
    stage_cfg = _stage_cfg(cfg, stage)
    min_vram = _parse_vram_bound(
        min_vram_gb if min_vram_gb is not None else stage_cfg.get("min_vram_gb", 0),
        field=f"tasks.{stage}.min_vram_gb",
        default=0,
        allow_none=False,
    )
    assert min_vram is not None
    raw_max = max_vram_gb if max_vram_gb is not None else stage_cfg.get("max_vram_gb")
    max_vram = _parse_vram_bound(
        raw_max,
        field=f"tasks.{stage}.max_vram_gb",
        default=None,
        allow_none=True,
    )
    if max_vram is not None and max_vram < min_vram:
        raise ValueError(
            f"tasks.{stage}: max_vram_gb ({max_vram}) < min_vram_gb ({min_vram})"
        )
    return min_vram, max_vram


def _stage_cfg(cfg: Mapping[str, Any], stage: str) -> Mapping[str, Any]:
    tasks = cfg.get("tasks") or {}
    if stage not in tasks:
        raise ValueError(f"unknown stage {stage!r}; expected one of {sorted(tasks)}")
    stage_cfg = tasks[stage] or {}
    if not isinstance(stage_cfg, Mapping):
        raise ValueError(f"tasks.{stage} must be a mapping")
    return stage_cfg


def _yaml_gpu_pin(cfg: Mapping[str, Any], stage: str) -> list[str] | None:
    stage_cfg = _stage_cfg(cfg, stage)
    pinned = stage_cfg.get("gpu")
    if pinned is None:
        return None
    if isinstance(pinned, str):
        return [pinned]
    if isinstance(pinned, list):
        if not all(isinstance(x, str) and x.strip() for x in pinned):
            raise ValueError(f"tasks.{stage}.gpu list must contain non-empty strings")
        return list(pinned)
    raise ValueError(f"tasks.{stage}.gpu must be null, a string, or a list of strings")

#############################################################

### RANKING (filter -> build LadderEntry -> sort -> truncate) 

#############################################################

@dataclass(frozen=True)
class LadderEntry:
    id: str
    cloud: str
    usd_per_hr: float
    vram_gb: int


def price_for(gpu: Mapping[str, Any], cloud: str) -> float | None:

    if cloud not in VALID_CLOUDS:
        raise ValueError(f"cloud must be one of {sorted(VALID_CLOUDS)}, got {cloud!r}")

    rates = gpu.get("usd_per_hr") or {}
    if not isinstance(rates, Mapping):
        raise ValueError(f"gpus[].usd_per_hr must be a mapping, got {rates!r}")

    raw = rates.get(cloud)  
    if raw is None:
        return None

    try:
        rate = float(raw)
    except ValueError:
        raise ValueError(f"GPU rate must be a number or null, got {raw!r}")
    return rate


def resolve_cloud(
    cfg: Mapping[str, Any],
    cloud_override: str | None = None,
    *,
    mounts_network_volume: bool = True,
) -> str:

    raw = cloud_override if cloud_override is not None else cfg.get("cloud", "secure")
    if not isinstance(raw, str):
        raise ValueError(f"cloud must be a string, got {raw!r}")

    cloud = raw.strip().lower()
    if cloud not in VALID_CLOUDS:
        raise ValueError(f"cloud must be one of {sorted(VALID_CLOUDS)}, got {raw!r}")

    if cloud == "community" and mounts_network_volume:
        raise ValueError(
            "cloud=community is incompatible with a network volume "
            "(network volumes require Secure Cloud). "
            "Use --runpod-cloud secure, or drop the volume for a community path."
        )
    return cloud


def resolve_datacenter(
    cfg: Mapping[str, Any],
    datacenter_override: str | None = None,
) -> str:

    if datacenter_override is not None:
        dc = datacenter_override.strip()
        if not dc:
            raise ValueError("--runpod-datacenter must be a non-empty string")
        return dc

    volume = cfg.get("volume") or {}
    if not isinstance(volume, Mapping):
        raise ValueError("volume config must be a mapping")

    dc = volume.get("datacenter")
    if not isinstance(dc, str) or not dc.strip():
        raise ValueError(
            "volume.datacenter is required when --runpod-datacenter is not set"
        )

    return dc.strip()


def resolve_gpu_ladder(
    cfg: Mapping[str, Any],
    stage: str,
    *,
    cloud: str,
    gpu_override: Sequence[str] | None = None,
    max_usd_per_hr: float | None = None,
    max_attempts: int | None = None,
    min_vram_gb: int | None = None,
    max_vram_gb: int | None = None,
    allowed_ids: set[str] | frozenset[str] | None = None,
) -> list[LadderEntry]:
    catalog = _catalog_by_id(cfg)
    cap, attempts = _retry_settings(cfg, max_usd_per_hr, max_attempts)

    def _reject_disallowed(gpu_id: str) -> None:
        if allowed_ids is not None and gpu_id not in allowed_ids:
            raise ValueError(
                f"GPU {gpu_id!r} is not a Flash-supported device name. "
                f"Use one of: {sorted(allowed_ids)}"
            )

    if gpu_override:
        ids = [g.strip() for g in gpu_override if isinstance(g, str) and g.strip()]
        if not ids:
            raise ValueError("gpu override list is empty")
        for gpu_id in ids:
            _reject_disallowed(gpu_id)
        entries = _entries_for_ids(catalog, ids, cloud, require_price=True)
        return entries[:attempts]

    pinned = _yaml_gpu_pin(cfg, stage)
    if pinned is not None:
        if not pinned:
            raise ValueError(f"tasks.{stage}.gpu pin list is empty")
        for gpu_id in pinned:
            _reject_disallowed(gpu_id)
        entries = _entries_for_ids(catalog, pinned, cloud, require_price=True)
        return entries[:attempts]

    min_vram, max_vram = _stage_vram_bounds(
        cfg, stage, min_vram_gb=min_vram_gb, max_vram_gb=max_vram_gb
    )
    ranked: list[LadderEntry] = []
    for gpu_id, gpu in catalog.items():
        if allowed_ids is not None and gpu_id not in allowed_ids:
            continue
        price = price_for(gpu, cloud)
        if price is None or price > cap:
            continue
        vram = int(gpu["vram_gb"])
        if vram < min_vram:
            continue
        if max_vram is not None and vram > max_vram:
            continue
        ranked.append(
            LadderEntry(id=gpu_id, cloud=cloud, usd_per_hr=price, vram_gb=vram)
        )
    ranked.sort(key=lambda e: (e.usd_per_hr, e.id))
    if not ranked:
        raise ValueError(
            f"no GPUs available for stage={stage!r} cloud={cloud!r} "
            f"min_vram_gb={min_vram} max_vram_gb={max_vram} max_usd_per_hr={cap}"
        )
    return ranked[:attempts]


def truncate_ladder(ladder: list[LadderEntry], max_attempts: int) -> list[LadderEntry]:
    if max_attempts < 1:
        raise ValueError(f"max_attempts must be >= 1, got {max_attempts}")
    return ladder[:max_attempts]
