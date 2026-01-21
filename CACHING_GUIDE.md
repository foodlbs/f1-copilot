# F1 Data Collection - Caching Guide

## How Caching Works Now

The system has **two levels of caching** to prevent re-downloading data:

### Level 1: FastF1 Cache
- **Location:** `cache/fastf1/`
- **What it caches:** Raw API responses from F1 servers
- **Managed by:** FastF1 library automatically
- **Benefit:** If download is interrupted, FastF1 doesn't re-fetch raw data

### Level 2: Processed Race Cache (NEW!)
- **Location:** `cache/f1_data/`
- **What it caches:**
  - Individual races: `race_2024_R1.json`, `race_2024_R2.json`, etc.
  - Complete seasons: `season_2024.json`, `season_2023.json`, etc.
- **Benefit:** Skip processing already-downloaded races

## Usage Examples

### 1. Normal Mode (Uses Cache)
```bash
cd src
python knowledge_base_builder.py
```
**Behavior:**
- ✅ Checks `cache/f1_data/` for existing race data
- ✅ Only downloads NEW or MISSING races
- ✅ Skips already-downloaded races
- **Time:** Fast! Only downloads what's needed

### 2. Force Re-download Mode
```bash
cd src
python knowledge_base_builder.py --force-redownload
```
**Behavior:**
- ❌ Ignores all cached race data
- ⬇️ Re-downloads EVERYTHING from scratch
- ⬇️ Re-processes all races
- **Time:** Slow (~40-50 minutes)
- **Use when:** Cache is corrupted or you want fresh data

### 3. Skip Collection (Use Only Cache)
```bash
cd src
python knowledge_base_builder.py --skip-collection
```
**Behavior:**
- ✅ Uses ONLY cached data (no downloads)
- ❌ Fails if cache is empty
- **Time:** Very fast (<1 minute)
- **Use when:** Data already collected, just want to re-process or re-ingest

### 4. Skip Vector DB Ingestion
```bash
cd src
python knowledge_base_builder.py --skip-ingestion
```
**Behavior:**
- ⬇️ Downloads race data (or uses cache)
- ❌ Does NOT create embeddings or ingest to Pinecone
- **Use when:** Just want to collect raw data first

### 5. Update with Latest Races Only
```bash
cd src
python knowledge_base_builder.py --update-only
```
**Behavior:**
- ✅ Checks latest race in cache
- ⬇️ Downloads only newer races
- ✅ Incremental update
- **Use when:** Keeping knowledge base current

### 6. Custom Start Year
```bash
cd src
python knowledge_base_builder.py --start-year 2020
```
**Behavior:**
- Downloads from 2020-present instead of 2017
- Faster initial build

### 7. Combine Flags
```bash
# Force re-download but skip vector DB
python knowledge_base_builder.py --force-redownload --skip-ingestion

# Update only latest races and skip ingestion
python knowledge_base_builder.py --update-only --skip-ingestion

# Start from 2020, use cache, just ingest to vector DB
python knowledge_base_builder.py --start-year 2020 --skip-collection
```

## What Gets Cached

### After First Run:
```
cache/
├── f1_data/
│   ├── race_2024_R1.json          # Bahrain GP 2024
│   ├── race_2024_R2.json          # Saudi Arabia GP 2024
│   ├── race_2024_R3.json          # Australia GP 2024
│   ├── ...
│   ├── season_2024.json           # All 2024 races combined
│   ├── season_2023.json           # All 2023 races combined
│   └── ...
└── fastf1/                        # FastF1 raw cache
    ├── 2024_1_R_session_info.pkl
    ├── 2024_1_R_timing_data.pkl
    └── ...
```

### Cache Sizes (Approximate):
- **Individual race:** ~50-100 KB each
- **Season cache:** ~2-4 MB per season
- **FastF1 cache:** ~5-10 MB per race weekend
- **Total (2017-2024):** ~2-3 GB

## Smart Caching Behavior

### Scenario 1: Fresh Install
```bash
python knowledge_base_builder.py
```
Result:
- Downloads all races from 2017-2024
- Caches each race individually
- Creates season cache files
- Time: ~40-50 minutes

### Scenario 2: Re-run After Interruption
```bash
# First run interrupted at 2022
python knowledge_base_builder.py
```
Result:
- ✅ Loads 2017-2022 from cache (instant)
- ⬇️ Downloads 2023-2024 only
- Time: ~10 minutes

### Scenario 3: Weekly Update
```bash
# New race happened this weekend
python knowledge_base_builder.py --update-only
```
Result:
- ✅ Loads existing data from cache
- ⬇️ Downloads only the new race
- ✅ Adds to vector database
- Time: <2 minutes

### Scenario 4: Corrupted Cache
```bash
# Some cached files are corrupted
python knowledge_base_builder.py --force-redownload
```
Result:
- ❌ Ignores all cache
- ⬇️ Re-downloads everything fresh
- Time: ~40-50 minutes

## Cache Management

### Clear All Cache
```bash
# Remove ALL cached data
rm -rf cache/f1_data/*
rm -rf cache/fastf1/*
```

### Clear Only Processed Cache (Keep FastF1)
```bash
# Remove processed race data but keep FastF1 raw cache
rm -rf cache/f1_data/*
```

### Clear Single Season
```bash
# Remove 2024 cache only
rm cache/f1_data/season_2024.json
rm cache/f1_data/race_2024_*.json
```

### Check Cache Status
```bash
# See what's cached
ls -lh cache/f1_data/

# Count cached races
ls cache/f1_data/race_*.json | wc -l

# See cache size
du -sh cache/
```

## Performance Comparison

| Scenario | First Run | With Cache | Force Redownload |
|----------|-----------|------------|------------------|
| 2017-2024 (full) | 45 min | 1 min | 45 min |
| 2020-2024 (recent) | 20 min | 30 sec | 20 min |
| Single season | 5 min | <1 sec | 5 min |
| Update (1 race) | 30 sec | <1 sec | 30 sec |

## Troubleshooting

### "Some races failed to download"
**Solution:** Re-run without `--force-redownload`
- The script will skip successful downloads
- Only retry failed races

### "Cache files are too old"
**Solution:** Use `--force-redownload` for specific years
```bash
# Clear 2024 cache manually, then run
rm cache/f1_data/*2024*.json
python knowledge_base_builder.py
```

### "Want to rebuild vector DB only"
**Solution:**
```bash
# Use cached data, just re-do ingestion
python knowledge_base_builder.py --skip-collection
```

## Best Practices

✅ **DO:**
- Use default mode (with cache) for normal operation
- Use `--update-only` for weekly/monthly updates
- Keep cache directory backed up
- Use `--skip-collection` when testing vector DB changes

❌ **DON'T:**
- Use `--force-redownload` unless necessary (wastes time)
- Delete cache without good reason
- Run multiple instances simultaneously (cache conflicts)

## Summary

**Normal workflow:**
1. First build: `python knowledge_base_builder.py` (45 min)
2. Weekly updates: `python knowledge_base_builder.py --update-only` (2 min)
3. Testing changes: `python knowledge_base_builder.py --skip-collection` (instant)

The caching system saves you **40+ minutes** on every re-run!
