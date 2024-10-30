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




-- Comment on the table
COMMENT ON TABLE gameweek_stats IS 'Gameweek statistics for Fantasy Premier League players, including performance metrics, expected stats, and match information. This dataset is derived from the vastaav merged_gws dataset.';

-- Comment on columns
COMMENT ON COLUMN gameweek_stats.name IS 'The name of the FPL player.';

COMMENT ON COLUMN gameweek_stats.position IS 'The playing position of the FPL player (e.g., Goalkeeper, Defender, Midfielder, Forward).';

COMMENT ON COLUMN gameweek_stats.team IS 'The team to which the player belongs during the gameweek.';

COMMENT ON COLUMN gameweek_stats.xP IS 'Expected points (xP) for the player, based on statistical models.';

COMMENT ON COLUMN gameweek_stats.assists IS 'Number of assists made by the player in the gameweek.';

COMMENT ON COLUMN gameweek_stats.bonus IS 'Bonus points awarded to the player for exceptional performance.';

COMMENT ON COLUMN gameweek_stats.bps IS 'Bonus Points System (BPS) score that determines bonus points awarded.';

COMMENT ON COLUMN gameweek_stats.clean_sheets IS 'Whether the player\'s team kept a clean sheet in the gameweek.';

COMMENT ON COLUMN gameweek_stats.creativity IS 'Creativity score, reflecting a player\'s contribution to attacking play.';

COMMENT ON COLUMN gameweek_stats.element IS 'Unique identifier for the player in the FPL database.';

COMMENT ON COLUMN gameweek_stats.expected_assists IS 'The number of assists the player was expected to provide, based on statistical modeling.';

COMMENT ON COLUMN gameweek_stats.expected_goal_involvements IS 'The total number of goal involvements (goals + assists) expected from the player.';

COMMENT ON COLUMN gameweek_stats.expected_goals IS 'The number of goals the player was expected to score, based on statistical modeling.';

COMMENT ON COLUMN gameweek_stats.expected_goals_conceded IS 'The number of goals the player\'s team was expected to concede.';

COMMENT ON COLUMN gameweek_stats.fixture IS 'Fixture ID, indicating the specific match in which the player participated.';

COMMENT ON COLUMN gameweek_stats.goals_conceded IS 'The actual number of goals conceded by the player\'s team.';

COMMENT ON COLUMN gameweek_stats.goals_scored IS 'The number of goals scored by the player.';

COMMENT ON COLUMN gameweek_stats.ict_index IS 'Index combining Influence, Creativity, and Threat (ICT), used to measure a player\'s overall contribution to a match.';

COMMENT ON COLUMN gameweek_stats.influence IS 'Influence score, reflecting the player\'s impact on their team\'s performance.';

COMMENT ON COLUMN gameweek_stats.kickoff_time IS 'The kickoff time of the match in which the player participated.';

COMMENT ON COLUMN gameweek_stats.minutes IS 'Total minutes played by the player during the gameweek.';

COMMENT ON COLUMN gameweek_stats.opponent_team IS 'ID of the opposing team in the fixture.';

COMMENT ON COLUMN gameweek_stats.own_goals IS 'Number of own goals scored by the player.';

COMMENT ON COLUMN gameweek_stats.penalties_missed IS 'Number of penalty kicks missed by the player.';

COMMENT ON COLUMN gameweek_stats.penalties_saved IS 'Number of penalties saved by the player (goalkeepers only).';

COMMENT ON COLUMN gameweek_stats.red_cards IS 'Number of red cards received by the player during the gameweek.';

COMMENT ON COLUMN gameweek_stats.round IS 'The round number of the gameweek in the FPL season.';

COMMENT ON COLUMN gameweek_stats.saves IS 'Number of saves made by the player (goalkeepers only).';

COMMENT ON COLUMN gameweek_stats.selected IS 'The number of managers who selected this player in their teams for the gameweek.';

COMMENT ON COLUMN gameweek_stats.starts IS 'Whether the player started the match.';

COMMENT ON COLUMN gameweek_stats.team_a_score IS 'The score of the away team in the match.';

COMMENT ON COLUMN gameweek_stats.team_h_score IS 'The score of the home team in the match.';

COMMENT ON COLUMN gameweek_stats.threat IS 'Threat score, reflecting the player\'s likelihood of scoring goals.';

COMMENT ON COLUMN gameweek_stats.total_points IS 'Total FPL points earned by the player during the gameweek.';

COMMENT ON COLUMN gameweek_stats.transfers_balance IS 'Net transfers for the player (transfers in minus transfers out).';

COMMENT ON COLUMN gameweek_stats.transfers_in IS 'Number of managers who transferred the player into their teams for the gameweek.';

COMMENT ON COLUMN gameweek_stats.transfers_out IS 'Number of managers who transferred the player out of their teams for the gameweek.';

COMMENT ON COLUMN gameweek_stats.value IS 'The player\'s FPL value (price) in the gameweek.';

COMMENT ON COLUMN gameweek_stats.was_home IS 'Boolean indicating whether the player\'s team was playing at home (true) or away (false).';

COMMENT ON COLUMN gameweek_stats.yellow_cards IS 'Number of yellow cards received by the player during the gameweek.';

COMMENT ON COLUMN gameweek_stats.GW IS 'The gameweek number (GW) within the FPL season.';
