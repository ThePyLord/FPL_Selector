# FPL Selector

AI-powered Fantasy Premier League decision support tool. Uses machine learning to predict player points and optimize team selection.

## Features

- **Point Predictions**: XGBoost-based model predicting next gameweek points
- **Transfer Suggestions**: Analyzes your squad and recommends improvements
- **Lineup Optimization**: Selects optimal starting XI and captain
- **Data Pipeline**: Automated ingestion from FPL API and Vaastav dataset

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/ThePyLord/FPL_Selector.git
cd FPL_Selector

# Install in development mode
pip install -e ".[dev]"
```

### Usage

```bash
# Update data from FPL API and Vaastav
fpl update

# View predictions for next gameweek
fpl predict --top 20

# Get transfer suggestions for your team
fpl transfers YOUR_TEAM_ID

# Get optimal lineup
fpl lineup YOUR_TEAM_ID

# Check data status
fpl status
```

### Finding Your Team ID

Your FPL team ID can be found in the URL when viewing your team on the FPL website:
`https://fantasy.premierleague.com/entry/YOUR_TEAM_ID/event/1`

## Project Structure

```
FPL_Selector/
├── src/fpl_selector/        # Main package
│   ├── cli.py               # Command-line interface
│   ├── data/                # Data collection & storage
│   ├── features/            # Feature engineering
│   ├── models/              # ML models (XGBoost predictor, optimizer)
│   └── services/            # High-level services
├── config/                  # Configuration files
├── data/                    # Data storage (Parquet files)
├── docs/                    # Documentation
│   └── PRD.md              # Product Requirements Document
└── tests/                   # Test suite
```

## Data Sources

- **[Vaastav FPL Dataset](https://github.com/vaastav/Fantasy-Premier-League)**: Historical gameweek data, updated semi-seasonally
- **[FPL API](https://fantasy.premierleague.com/api/)**: Current player info, fixtures, team data

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run linter
ruff check src/

# Type checking
mypy src/fpl_selector/
```

## Roadmap



### MVP (Target: Matchday 30)
- [x] Data pipeline (collectors, storage, validators)
- [ ] Feature engineering
- [ ] XGBoost model training
- [ ] CLI commands (`predict`, `transfers`, `lineup`)
- [ ] Documentation

### Post-MVP
- [ ] Web interface
- [ ] Multi-gameweek planning
- [ ] Chip optimization (wildcard, bench boost, etc.)
- [ ] Live gameweek tracking

## License

MIT

## Author

Victor Nnah