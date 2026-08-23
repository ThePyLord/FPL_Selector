# FPL Selector - Product Requirements Document (PRD)

## Document Information
- **Version**: 1.0
- **Created**: January 26, 2026
- **Target Delivery**: Matchday 30 (approximately 4 weeks)
- **Status**: Active

---

## 1. Executive Summary

### 1.1 Problem Statement
Fantasy Premier League (FPL) managers struggle to make optimal weekly decisions due to:
- **Information overload**: 600+ players, 20 teams, complex statistics
- **Time constraints**: Limited time to analyze form, fixtures, and injuries
- **Cognitive bias**: Emotional attachments to players cloud judgment
- **Optimization complexity**: Budget constraints, formation rules, and transfer limits

### 1.2 Solution
**FPL Selector** is a CLI-based decision support tool that:
1. Predicts player points for the next gameweek
2. Recommends optimal transfers within budget
3. Suggests starting XI and captain picks

### 1.3 Success Metrics (MVP)
| Metric | Target | Measurement |
|--------|--------|-------------|
| Prediction Accuracy | >60% directional accuracy | Predicted vs actual points ranking |
| Model Refresh | Weekly before deadline | Automated pipeline completion |
| User Commands | 3 core commands working | `predict`, `transfers`, `lineup` |
| Data Freshness | <24 hours before gameweek | Time since last data pull |

---

## 2. Scope Definition

### 2.1 In Scope (MVP - Matchday 30)
| Feature | Priority | Description |
|---------|----------|-------------|
| **Data Pipeline** | P0 | Automated ingestion from Vaastav dataset + FPL API |
| **Point Prediction** | P0 | XGBoost model predicting next GW points |
| **Transfer Suggestions** | P0 | Top 5 transfer recommendations |
| **Starting XI** | P1 | Optimal lineup from current squad |
| **Captain Pick** | P1 | Recommended captain and vice-captain |
| **CLI Interface** | P0 | Simple command-line interface |

### 2.2 Out of Scope (Post-MVP)
- Web/Mobile UI
- Multi-gameweek planning (chip strategy)
- Social features (league comparisons)
- Injury probability modeling
- Live gameweek tracking
- Player price prediction

---

## 3. User Stories

### 3.1 Core User Journey
```
As an FPL manager,
I want to get data-driven recommendations before the gameweek deadline,
So that I can maximize my team's points without hours of research.
```

### 3.2 MVP User Stories

| ID | Story | Acceptance Criteria |
|----|-------|---------------------|
| US-01 | As a user, I want to see predicted points for all players | `fpl predict` shows sorted player predictions |
| US-02 | As a user, I want to know which transfers to make | `fpl transfers --team-id <id>` shows top 5 transfer suggestions |
| US-03 | As a user, I want to know my optimal starting XI | `fpl lineup --team-id <id>` shows recommended lineup |
| US-04 | As a user, I want captain recommendations | `fpl captain --team-id <id>` shows captain picks |
| US-05 | As a user, I want to update data before making decisions | `fpl update` refreshes all data sources |

---

## 4. Data Requirements

### 4.1 Data Sources

| Source | Data | Update Frequency | Priority |
|--------|------|------------------|----------|
| **Vaastav GitHub** | Historical gameweek data | Weekly (after GW ends) | P0 |
| **FPL Bootstrap API** | Current player info, prices, teams | Daily | P0 |
| **FPL Fixtures API** | Upcoming fixtures, difficulty | Daily | P0 |
| **FPL Entry API** | User's current team | On-demand | P0 |

### 4.2 Core Data Entities

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│     Players     │     │   Gameweeks     │     │    Fixtures     │
├─────────────────┤     ├─────────────────┤     ├─────────────────┤
│ id (PK)         │     │ player_id (FK)  │     │ fixture_id (PK) │
│ name            │◄────│ gameweek        │     │ gameweek        │
│ team_id (FK)    │     │ points          │     │ team_h (FK)     │
│ position        │     │ minutes         │     │ team_a (FK)     │
│ price           │     │ goals           │     │ kickoff_time    │
│ selected_by     │     │ assists         │     │ difficulty_h    │
└─────────────────┘     │ xG, xA, etc.    │     │ difficulty_a    │
        │               └─────────────────┘     └─────────────────┘
        │                                               │
        ▼                                               │
┌─────────────────┐                                     │
│     Teams       │◄────────────────────────────────────┘
├─────────────────┤
│ id (PK)         │
│ name            │
│ short_name      │
│ strength        │
└─────────────────┘
```

### 4.3 Feature Requirements

| Category | Features | Used For |
|----------|----------|----------|
| **Form** | Rolling avg (3, 5 GW) of points, goals, assists | Recent performance |
| **Fixture** | Opponent strength, home/away, fixture difficulty | Context adjustment |
| **Value** | Price, ownership %, transfer trends | Market signals |
| **Expected Stats** | xG, xA, xGI per 90 | Underlying quality |
| **Minutes** | Minutes played, starts, availability | Playing time prediction |

---

## 5. Technical Requirements

### 5.1 Architecture Overview (MVP)

```
┌────────────────────────────────────────────────────────────────┐
│                         CLI Interface                          │
│                    (Typer / Click)                             │
└────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────────────┐
│                      Application Layer                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │  Predictor   │  │  Optimizer   │  │  Recommender │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
└────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────────────┐
│                        ML Pipeline                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │FeatureEngine │  │  XGBModel    │  │  Evaluator   │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
└────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────────────┐
│                        Data Layer                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │  Collectors  │  │   Storage    │  │  Validators  │          │
│  │ (API/GitHub) │  │  (Parquet)   │  │              │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
└────────────────────────────────────────────────────────────────┘
```

### 5.2 Technology Stack (MVP)

| Layer | Technology | Rationale |
|-------|------------|-----------|
| **CLI** | Typer | Modern, type-safe CLI framework |
| **Data** | Polars | Fast DataFrame operations |
| **Storage** | Parquet + SQLite | Simple, portable, no server needed |
| **ML** | XGBoost | Proven for tabular data, fast training |
| **Config** | YAML | Human-readable configuration |
| **Testing** | Pytest | Standard Python testing |

### 5.3 Performance Requirements

| Requirement | Target | Notes |
|-------------|--------|-------|
| Data refresh | <2 minutes | Full pipeline run |
| Prediction generation | <10 seconds | All 600+ players |
| Transfer optimization | <30 seconds | With constraints |
| Model retraining | <10 minutes | On consumer laptop |

---

## 6. Constraints & Assumptions

### 6.1 Constraints
- **Budget**: $0 infrastructure (local-only for MVP)
- **Timeline**: 4 weeks to Matchday 30
- **Data**: Dependent on external APIs (may have rate limits)
- **Compute**: Must run on standard laptop (no GPU required)

### 6.2 Assumptions
- Vaastav dataset continues to be updated weekly (NOT True anymore as of 2026/27 season)
- FPL API remains stable and accessible
- User has Python 3.10+ installed
- User has internet connection for data updates

---

## 7. Risks & Mitigations

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| FPL API changes | High | Low | Cache responses, graceful degradation |
| Poor model accuracy | Medium | Medium | Ensemble approach, regular retraining |
| Vaastav dataset delay | Medium | Low | Fallback to API-only mode |
| Feature scope creep | High | High | Strict MVP scope, backlog prioritization |

---

## 8. MVP Delivery Plan

### 8.1 Sprint Breakdown (4 weeks)

#### Week 1: Data Foundation
- [ ] Clean up existing code, remove dead files
- [ ] Implement robust data collectors (Vaastav + FPL API)
- [ ] Set up Parquet storage layer
- [ ] Create data validation pipeline
- [ ] Write data pipeline tests

#### Week 2: Feature Engineering & Model
- [ ] Implement core feature engineering
- [ ] Train XGBoost model with cross-validation
- [ ] Create model evaluation framework
- [ ] Implement model serialization/loading
- [ ] Write model tests

#### Week 3: Application Layer & CLI
- [ ] Build prediction service
- [ ] Build transfer recommendation engine
- [ ] Build lineup optimizer
- [ ] Implement Typer CLI
- [ ] Integration testing

#### Week 4: Polish & Deploy
- [ ] End-to-end testing
- [ ] Documentation (README, CLI help)
- [ ] Bug fixes
- [ ] Package for distribution (pip install)
- [ ] First real gameweek test

### 8.2 Definition of Done (MVP)
- [ ] `fpl update` - Refreshes all data
- [ ] `fpl predict` - Shows next GW predictions
- [ ] `fpl transfers --team-id <id>` - Shows transfer suggestions
- [ ] `fpl lineup --team-id <id>` - Shows optimal starting XI
- [ ] All tests passing
- [ ] README with installation and usage instructions

---

## 9. File Structure (Target)

```
FPL_Selector/
├── README.md
├── pyproject.toml              # Package definition
├── config/
│   ├── config.yaml             # Main config
│   └── features.yaml           # Feature definitions
├── data/
│   ├── raw/                    # Raw data files
│   │   ├── gameweeks/          # Parquet files per season
│   │   └── fixtures/
│   ├── processed/              # Feature-engineered data
│   └── models/                 # Trained models
├── src/
│   └── fpl_selector/
│       ├── __init__.py
│       ├── cli.py              # Typer CLI
│       ├── data/
│       │   ├── __init__.py
│       │   ├── collectors.py   # API/GitHub collectors
│       │   ├── storage.py      # Parquet I/O
│       │   └── validators.py   # Data validation
│       ├── features/
│       │   ├── __init__.py
│       │   └── engineering.py  # Feature transformations
│       ├── models/
│       │   ├── __init__.py
│       │   ├── predictor.py    # XGBoost wrapper
│       │   └── optimizer.py    # Team optimization
│       └── services/
│           ├── __init__.py
│           ├── predictions.py  # Prediction service
│           ├── transfers.py    # Transfer recommendations
│           └── lineup.py       # Lineup optimization
└── tests/
    ├── conftest.py
    ├── test_collectors.py
    ├── test_features.py
    └── test_models.py
```

---

## 10. Success Criteria

### 10.1 MVP Launch (Matchday 30)
- [ ] CLI tool installable via `pip install -e .`
- [ ] All 4 core commands functional
- [ ] Prediction accuracy measured on GW 26-29
- [ ] Documentation complete

### 10.2 Post-MVP Metrics (Track for 4 weeks)
- Prediction directional accuracy (target: >60%)
- User satisfaction with recommendations
- Model performance vs FPL average

---

## Appendix A: Current Codebase Assessment

### Files to Keep (Refactor)
| File | Status | Action |
|------|--------|--------|
| `db_manager.py` | Partial | Refactor to use Parquet, keep for optional PostgreSQL |
| `feature_engineering.py` | Good base | Clean up, modularize |
| `ingestion/collector.py` | Good base | Refactor, add error handling |

### Files to Remove/Archive
| File | Reason |
|------|--------|
| `pipeline.py` | Empty shell |
| `optimize_team.py` | Empty shell |
| `ml_model.py` | Superseded by new structure |
| `xgb_model.py` | Script-style, refactor into module |
| `lstm_model.py` | Out of MVP scope |
| `f_eng.py` | Duplicate of feature_engineering |
| `league.py` | Out of MVP scope |
| `top_performers.py` | Out of MVP scope |
| `test.py` | Not a real test file |
| `update_db.py` | Superseded |

---

## Appendix B: CLI Command Specifications

### `fpl update`
```bash
$ fpl update
✓ Fetching current season data from Vaastav...
✓ Fetching player data from FPL API...
✓ Fetching fixture data...
✓ Updating local storage...
✓ Data updated successfully (took 45s)
  Last gameweek: 25
  Players: 642
  Next deadline: Sat 15:00
```

### `fpl predict [--position POS] [--top N]`
```bash
$ fpl predict --top 10
Next Gameweek Predictions (GW 26)

Rank | Player           | Team | Pos | Pred | Form | Fixture
-----|------------------|------|-----|------|------|----------
1    | Mohamed Salah    | LIV  | MID | 8.2  | 7.4  | BOU (H)
2    | Erling Haaland   | MCI  | FWD | 7.8  | 8.1  | EVE (A)
...
```

### `fpl transfers --team-id <id> [--free-transfers N] [--budget B]`
```bash
$ fpl transfers --team-id 123456
Analyzing your team...

Suggested Transfers (1 free, £1.2m ITB):

OUT                  | IN                   | Δ Points | Cost
---------------------|----------------------|----------|------
Watkins (5.8, AVL)   | Isak (8.2, NEW)     | +2.4     | -£2.4
```

### `fpl lineup --team-id <id>`
```bash
$ fpl lineup --team-id 123456
Optimal Starting XI (GW 26)

Formation: 3-4-3

GK:  Raya (7.2)
DEF: Gabriel (5.1) | Saliba (5.4) | Van Dijk (4.8)
MID: Salah (8.2) | Palmer (6.8) | Saka (6.1) | Rogers (4.2)
FWD: Haaland (7.8) | Isak (6.4) | Watkins (4.9)

Captain: Salah (8.2) | Vice: Haaland (7.8)

Bench: Martinez, Lewis, Wharton, Archer
```
