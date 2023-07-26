import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency


class PlayerPerformance:
    player_name = None
    player_country = None

    def __init__(self, path="odis_male_csv2", mark_fab_four=True):
        self.path = path
        self.mark_fab_four = mark_fab_four
        self.all_matches = self._load_dataset()

    def player_luck_comparision(self, **kwargs):
        players = self.get_runs_by_batsmen(**kwargs)
        return pd.DataFrame(
            [
                self.chi_sq_player_century_and_game_result(
                    player_name=player.batter, runs=player.runs
                )
                for _, player in players.iterrows()
            ]
        )

    def plot_team_win_pct(self, player_name):
        self._preprocess_for_player(player_name)

        featured = self.engineered_features

        score_bins = [0, 20, 50, 85, 95, 100, 115, 150, 205]
        featured["binned_score"] = pd.cut(
            featured[f"{self.player_last_name}_score"], score_bins, include_lowest=True
        )

        player_career = featured.loc[featured.batting_team.isin(self.player_country)]

        team_win_pct = (
            player_career.groupby(["binned_score"])
            .agg(win=("team_won", "sum"), total_games=("match_id", "nunique"))
            .reset_index()
            .assign(win_pct=lambda df: df.win / df.total_games)
        )

        team_win_pct.binned_score = team_win_pct.binned_score.astype("str")

        team_win_pct.plot(x="binned_score", y="win_pct", kind="bar")
        plt.title(
            f"{self.player_name} Score v {', '.join(self.player_country)} Win Prob"
        )
        return plt.show(), team_win_pct

    def plot_team_win_pct_cumsum(self, player_name):
        self._preprocess_for_player(player_name)

        featured = self.engineered_features

        distinct_scores = (
            featured[f"{self.player_last_name}_score"].dropna().sort_values().unique()
        )

        win_pct_profile = []
        for score in distinct_scores:
            wins = featured.loc[
                featured[f"{self.player_last_name}_score"] >= score, "team_won"
            ].sum()
            games = featured.loc[
                featured[f"{self.player_last_name}_score"] >= score, "match_id"
            ].nunique()
            win_pct_profile.append(
                {"score": score, "prob": wins / games, "games": games}
            )

        win_pct_profile = pd.DataFrame(win_pct_profile)
        win_pct_profile.plot(x="score", y="prob", kind="line")
        plt.title(
            f"{self.player_name} Score greater then v {', '.join(self.player_country)} Win Prob"
        )

        return plt.show(), win_pct_profile

    def chi_sq_player_century_and_game_result(self, player_name, runs):
        self._preprocess_for_player(player_name)

        player_career = self.engineered_features.loc[
            self.engineered_features.batting_team.isin(self.player_country)
        ]

        player_career[f"{self.player_last_name}_century"] = (
            player_career[f"{self.player_last_name}_score"] >= 100
        )
        contingency_table = pd.crosstab(
            player_career[f"{self.player_last_name}_century"],
            player_career["team_won"],
        )

        chi2, p, dof, ex = chi2_contingency(contingency_table)

        contingency_table = contingency_table.reindex([False, True], fill_value=0)

        correlation_coefficient = player_career[f"{self.player_last_name}_score"].corr(
            player_career["team_won"]
        )

        century_precision = (
            contingency_table.loc[True][True] / contingency_table.loc[True].sum()
        )

        if (
            player_name
            in [
                "KS Williamson",
                "SPD Smith",
                "V Kohli",
                "JE Root",
            ]
            and self.mark_fab_four
        ):
            mark = "fab_four"
        elif player_name == "SR Tendulkar":
            mark = "marked"
        else:
            mark = "unmarked"

        try:
            return {
                "player_name": player_name,
                "runs": runs,
                "p": p,
                "chi2": chi2,
                "correlation": correlation_coefficient,
                "sample": len(player_career),
                "century_in_winning_cause": contingency_table.loc[True][True],
                "century_in_losing_cause": contingency_table.loc[True][False],
                "century_precision": century_precision
                if not np.isnan(century_precision)
                else 0,
                "century_accuracy": contingency_table.loc[True].sum()
                / len(player_career),
                "marked": mark,
            }
        except:
            print(contingency_table)
            return {}

    def get_runs_by_batsmen(self, n=5, **kwargs):
        players = self.all_matches.groupby(["striker"])

        career = (
            players.agg(
                batter=("striker", "first"),
                country=("batting_team", "first"),
                matches=("match_id", "nunique"),
                runs=("runs_off_bat", "sum"),
            )
            .reset_index(drop=True)
            .sort_values("runs", ascending=False)
        )

        tendulkar_career = career.query("batter == 'SR Tendulkar'")

        if kwargs.get("query"):
            return pd.concat([tendulkar_career, career.query(kwargs["query"])])

        return pd.concat([tendulkar_career, career.head(n)])

    def _preprocess_for_player(self, player_name):
        self.player_name = player_name
        self.player_last_name = player_name.split(" ")[-1].lower()

        matches_played_by_player = self.all_matches.loc[
            self.all_matches["striker"] == self.player_name, "match_id"
        ].unique()

        self.player_country = self.all_matches.loc[
            self.all_matches["striker"] == self.player_name, "batting_team"
        ].unique()

        played_matches = self.all_matches[
            self.all_matches.match_id.isin(matches_played_by_player)
        ]

        self._engineer_features(played_matches)

    def _engineer_features(self, df):
        innings = df.groupby(["start_date", "match_id", "innings"])

        features = pd.DataFrame(
            [self._innings_features(inning).iloc[0] for (_, _, _), inning in innings]
        )

        engineered_features = []
        matches = features.groupby(["start_date", "match_id"])
        self.engineered_features = features
        for (_, _), match in matches:
            scores = (
                match.groupby(["batting_team"])
                .agg(team_score=("team_score", "sum"))
                .reset_index()
            )
            winning_team = scores.loc[scores.team_score.idxmax(), "batting_team"]
            match["team_won"] = winning_team in self.player_country

            engineered_features.append(match)

        self.engineered_features = pd.concat(engineered_features)

    def _innings_features(self, df):
        ## Sorting
        df.sort_values(
            ["start_date", "match_id", "innings", "over", "ball_no"], inplace=True
        )
        ## team features
        df["total_runs_gained"] = df["runs_off_bat"] + df["extras"]
        df["score_board"] = df["total_runs_gained"].cumsum()

        df["dismissal"] = np.where(df["wicket_type"].notnull(), True, False)
        df["wickets_lost"] = df.dismissal.cumsum()
        df["team_score"] = df.score_board.max()
        df["five_wickets_lost_score"] = df[df["wickets_lost"] < 5].score_board.max()
        df["first_10_over_wickets_lost"] = df[df["over"] < 10].wickets_lost.max()

        if df.batting_team.unique() not in self.player_country:
            return df

        # player features
        df[f"{self.player_last_name}_score"] = df[
            df["striker"] == self.player_name
        ].runs_off_bat.sum()
        df["rest_team_score"] = (
            df.score_board.max() - df[f"{self.player_last_name}_score"].iloc[0]
        )

        player_out = (df["striker"] == self.player_name) & (df["wicket_type"].notnull())

        player_out_over = df.loc[player_out, "over"]
        df[f"{self.player_last_name}_out_over"] = (
            player_out_over.values[0] if player_out_over.size > 0 else 99
        )

        player_out_team_score = df.loc[player_out, "score_board"]
        df[f"team_score_when_{self.player_last_name}_got_out"] = (
            player_out_team_score.values[0]
            if player_out_team_score.size > 0
            else df.team_score.max()
        )

        player_out_team_wickets = df.loc[player_out, "wickets_lost"]
        df[f"team_wickets_when_{self.player_last_name}_got_out"] = (
            player_out_team_wickets.values[0]
            if player_out_team_wickets.size > 0
            else 99
        )

        df["pending_work_for_team"] = (
            df.score_board.max()
            - df[f"team_score_when_{self.player_last_name}_got_out"].values[0]
        )

        return df

    def _read_csv(self, path):
        try:
            return pd.read_csv(path)
        except pd.errors.ParserError:
            print(f"Faulty file -> {path}")

    def _load_dataset(self):
        matches = [
            f"{self.path}/{file}"
            for file in os.listdir(path=self.path)
            if not file.__contains__("info")
        ]
        df_matches = pd.concat([self._read_csv(m) for m in matches])
        return self._data_wrangling(df_matches)

    def _data_wrangling(self, df):
        df["over"], df["ball_no"] = divmod(df["ball"] * 10, 10)
        return df.assign(
            start_date=pd.to_datetime(df.start_date, format="%Y-%m-%d"),
            over=df.over.astype(int),
            ball_no=df.ball_no.astype(int),
            noballs=df.noballs.replace(np.NaN, 0).astype(int),
            wides=df.wides.replace(np.NaN, 0).astype(int),
            legbyes=df.legbyes.replace(np.NaN, 0).astype(int),
            penalty=df.penalty.replace(np.NaN, 0).astype(int),
        )
