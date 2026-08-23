-- Active: 1728268353503@@127.0.0.1@5432@fpl_db@public
CREATE TABLE players (

);

CREATE TABLE transfers (
	"time" TIMESTAMP NOT NULL,
	player_in INT NOT NULL,
	amount_in INT NOT NULL,
	amount_out INT NOT NULL,
	player_out INT NOT NULL,
	entry_id INT NOT NULL, -- ID of the FPL manager who made the transfer
	PRIMARY KEY ("time", player_in, player_out)
);

-- League standings for each gameweek
CREATE TABLE standings (
	id INT NOT NULL PRIMARY KEY, -- Unique ID
	gw_total INT NOT NULL, -- Total points for the gameweek
	player_name TEXT NOT NULL, -- Name of the FPL manager
	rank INT NOT NULL, -- Overall rank
	last_rank INT NOT NULL, -- Last week's rank
	total INT NOT NULL, -- Total points acquired in the season
	rank_sort INT NOT NULL,
	entry_id INT NOT NULL REFERENCES transfers(entry_id), -- ID of the FPL manager
	entry_name TEXT NOT NULL -- Name of the FPL team
);

CREATE TABLE gameweek_stats (
	name TEXT NOT NULL,
	position TEXT NOT NULL,
	team TEXT NOT NULL,
	xP REAL,
	assists INT,
	bonus INT,
	bps INT,
	clean_sheets INT,
	creativity REAL,
	element INT,
	expected_assists REAL,
	expected_goal_involvements REAL,
	expected_goals REAL,
	expected_goals_conceded REAL,
	fixture INT,
	goals_conceded INT,
	goals_scored INT,
	ict_index REAL,
	influence REAL,
	kickoff_time TIMESTAMP WITH TIME ZONE NOT NULL,
	minutes INT,
	opponent_team INT,
	own_goals INT,
	penalties_missed INT,
	penalties_saved INT,
	red_cards INT,
	round INT,
	saves INT,
	selected INT,
	starts INT,
	team_a_score INT,
	team_h_score INT,
	threat REAL,
	total_points INT,
	transfers_balance INT,
	transfers_in INT,
	transfers_out INT,
	value REAL,
	was_home BOOLEAN,
	yellow_cards INT,
	GW INT,
	PRIMARY KEY (name, GW, kickoff_time)
);


-- Teams table
CREATE TABLE IF NOT EXISTS fpl_teams (
	code INT NOT NULL PRIMARY KEY,
	name TEXT NOT NULL,
	id INT NOT NULL,
)