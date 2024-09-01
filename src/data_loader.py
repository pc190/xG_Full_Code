import pandas as pd
from statsbombpy import sb
import os

class DataLoader:
    def __init__(self, base_dir=None):
        # Set the base directory to the project's root
        self.base_dir = base_dir or os.path.dirname(os.getcwd())
        self.three_sixty_data_dir = os.path.join(self.base_dir, 'data', 'three-sixty')
        
        # Initialise empty shots dataframe
        self.shots_df = pd.DataFrame()

    def get_competitions(self):
        # Get available competitions summary
        return sb.competitions()

    def get_matches(self):
        # Return matches for used competitions
        matches_euro2024 = sb.matches(competition_id=55, season_id=282)
        matches_wc2022 = sb.matches(competition_id=43, season_id=106)
        matches_euro2020 = sb.matches(competition_id=55, season_id=43)
        matches_bundes_24 = sb.matches(competition_id=9, season_id=281)
        
        # Concatenate matches into a single data frame
        all_matches = pd.concat([matches_euro2024, matches_wc2022, matches_euro2020, matches_bundes_24], ignore_index=True)
        return all_matches

    def load_match_data(self, all_matches):
        # Loop through each match in the all_matches dataframe
        for m_id in all_matches['match_id']:
            # Get the standard event data for the match
            match_events_df = sb.events(match_id=m_id)

            # Construct the path to the 360 data for this match
            match_360_path = os.path.join(self.three_sixty_data_dir, f'{m_id}.json')

            # If the match does not have available 360 event data, continue to the next iteration of the loop
            if os.path.exists(match_360_path):
                match_360_df = pd.read_json(match_360_path)
            else:
                print(f"Could not find event data for match: {m_id}")
                continue

            # Join the standard match event data with the 360 event data
            df = pd.merge(left=match_events_df, right=match_360_df, left_on='id', right_on='event_uuid', how='left')

            # Filter the dataframe to include shots only
            shots_df_match = df[df['type'] == 'Shot']

            # Append the match shots to shots_df
            self.shots_df = pd.concat([self.shots_df, shots_df_match], ignore_index=True)

    def process_shots_data(self, all_matches):
        # Split the location into separate columns for x and y
        self.shots_df[['x_shot', 'y_shot']] = pd.DataFrame(self.shots_df['location'].tolist(), index=self.shots_df.index)
        
        # Add binary goal outcome
        self.shots_df['goal'] = (self.shots_df['shot_outcome'] == 'Goal').astype(int)

        # Remove fully null columns
        self.shots_df.dropna(axis=1, how='all', inplace=True)

        # Join relevant match info for each shot
        match_info = all_matches[['match_id', 'competition', 'home_team', 'away_team', 'match_date']]
        self.shots_df = pd.merge(left=self.shots_df, right=match_info, left_on='match_id', right_on='match_id', how='left')

        # Only true values appear in some referenced StatsBomb columns. Replace the null values with False.
        self.shots_df['under_pressure'] = self.shots_df['under_pressure'].fillna(False)
        self.shots_df['shot_open_goal'] = self.shots_df['shot_open_goal'].fillna(False)
        self.shots_df['shot_one_on_one'] = self.shots_df['shot_one_on_one'].fillna(False)
        self.shots_df['shot_first_time'] = self.shots_df['shot_first_time'].fillna(False)

    def save_shots_data(self, filename='shots_df.csv'):
        # Path to save the shots_df.csv file in the data directory
        output_path = os.path.join(self.base_dir, 'data', filename)

        # Save the DataFrame to the specified path
        self.shots_df.to_csv(output_path, index=False)

    def run(self):
        # Run the entire data loading and processing pipeline
        comps = self.get_competitions()
        all_matches = self.get_matches()
        self.load_match_data(all_matches)
        self.process_shots_data(all_matches)
        self.save_shots_data()

if __name__ == "__main__":
    # Create an instance of MatchDataLoader and execute the workflow
    data_loader = DataLoader()
    data_loader.run()
