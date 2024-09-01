
#Import modules
import pandas as pd
import matplotlib.pyplot as plt
from statsbombpy import sb
from mplsoccer import Pitch, VerticalPitch
import os
import ast
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score, roc_auc_score, brier_score_loss, log_loss
import numpy as np

class RathkeAnalysis:
    def __init__(self, base_dir=None):
        # Set the base directory to the project's root
        self.base_dir = base_dir or os.path.dirname(os.getcwd())
        self.shots_df = None

    def load_data(self, filename='shots_df.csv'):
        # Read in the shots dataframe
        file_path = os.path.join(self.base_dir, 'data', filename)
        self.shots_df = pd.read_csv(file_path)

    def preprocess_data(self):
        # For the purpose of this model, we want to use penalties only
        self.shots_df = self.shots_df[self.shots_df['shot_type'] != 'Penalty']

        # Fill null shot freezeframes with an empty string and convert the freeze_frame column back to a list of dictionaries
        self.shots_df['freeze_frame'] = self.shots_df['freeze_frame'].fillna('[]')
        self.shots_df['freeze_frame'] = self.shots_df['freeze_frame'].apply(ast.literal_eval)

        # Skip empty freezeframes
        self.shots_df = self.shots_df[self.shots_df['freeze_frame'].apply(lambda x: len(x) > 0)]

        # Filter out rows where there is no goalkeeper who is not a teammate
        self.shots_df = self.shots_df[self.shots_df['freeze_frame'].apply(self.has_non_teammate_goalkeeper)]

    @staticmethod
    def has_non_teammate_goalkeeper(freeze_frame):
        # For every player in the freezeframe
        for player in freeze_frame:
            # Return true if there is a player tagged as a goalkeeper and not a teammate
            if player.get('keeper', False) and not player.get('teammate', False):
                return True
        return False

    def visualize_zones(self):
        # Create the pitch with similar specifications
        pitch = VerticalPitch(pitch_color='#aabb97', line_color='white',
                              stripe_color='#c2d59d', stripe=True, pitch_type='statsbomb', axis=True, label=True, half=True)
        # Set up the figure
        fig, ax = pitch.draw(figsize=(8, 12))

        # Draw the sections as in the first image
        self.draw_zones(ax)

        # Adding labels to the centre of each zone
        self.add_zone_labels(ax)

        plt.show()

    def draw_zones(self, ax):
        # Zone 1
        ax.add_patch(plt.Rectangle((36, 114), 8, 6, fill=None, edgecolor='black', linewidth=2))
        # Zone 2
        ax.add_patch(plt.Rectangle((30, 114), 6, 6, fill=None, edgecolor='black', linewidth=2))
        ax.add_patch(plt.Rectangle((44, 114), 6, 6, fill=None, edgecolor='black', linewidth=2))
        # Zone 3
        ax.add_patch(plt.Rectangle((36, 102), 8, 12, fill=None, edgecolor='black', linewidth=2))
        # Zone 4
        ax.add_patch(plt.Rectangle((18, 114), 12, 6, fill=None, edgecolor='black', linewidth=2))
        ax.add_patch(plt.Rectangle((50, 114), 12, 6, fill=None, edgecolor='black', linewidth=2))
        # Zone 5
        ax.add_patch(plt.Rectangle((18, 102), 18, 12, fill=None, edgecolor='black', linewidth=2))
        ax.add_patch(plt.Rectangle((44, 102), 18, 12, fill=None, edgecolor='black', linewidth=2))
        # Zone 6
        ax.add_patch(plt.Rectangle((0, 81), 80, 21, fill=None, edgecolor='black', linewidth=2))
        # Zone 7
        ax.add_patch(plt.Rectangle((0, 102), 18, 18, fill=None, edgecolor='black', linewidth=2))
        ax.add_patch(plt.Rectangle((62, 102), 18, 18, fill=None, edgecolor='black', linewidth=2))
        # Zone 8
        ax.add_patch(plt.Rectangle((0, 50), 80, 31, fill=None, edgecolor='black', linewidth=2))

    def add_zone_labels(self, ax):
        # Adding labels to the centre of each zone
        ax.text(40, 117, '1', ha='center', va='center', fontsize=12, color='black')
        ax.text(33, 117, '2', ha='center', va='center', fontsize=12, color='black')
        ax.text(47, 117, '2', ha='center', va='center', fontsize=12, color='black')
        ax.text(40, 108, '3', ha='center', va='center', fontsize=12, color='black')
        ax.text(24, 117, '4', ha='center', va='center', fontsize=12, color='black')
        ax.text(56, 117, '4', ha='center', va='center', fontsize=12, color='black')
        ax.text(27, 108, '5', ha='center', va='center', fontsize=12, color='black')
        ax.text(53, 108, '5', ha='center', va='center', fontsize=12, color='black')
        ax.text(40, 90, '6', ha='center', va='center', fontsize=12, color='black')
        ax.text(9, 111, '7', ha='center', va='center', fontsize=12, color='black')
        ax.text(71, 111, '7', ha='center', va='center', fontsize=12, color='black')
        ax.text(40, 70, '8', ha='center', va='center', fontsize=12, color='black')

    def determine_zones(self):
        # Function to determine the zone of the shot according to Rathke's method
        self.shots_df['zone'] = self.shots_df.apply(lambda row: self.determine_zone(row['y_shot'], row['x_shot']), axis=1)

    @staticmethod
    def determine_zone(x, y):
        if 36 <= x < 44 and 114 <= y <= 120:
            return 1
        elif (30 <= x < 36 and 114 <= y <= 120) or (44 <= x < 50 and 114 <= y <= 120):
            return 2
        elif 36 <= x < 44 and 102 <= y < 114:
            return 3
        elif (18 <= x < 30 and 114 <= y <= 120) or (50 <= x < 62 and 114 <= y <= 120):
            return 4
        elif (18 <= x < 36 and 102 <= y < 114) or (44 <= x < 62 and 102 <= y < 114):
            return 5
        elif 0 <= x < 80 and 81 <= y < 102:
            return 6
        elif (0 <= x < 18 and 102 <= y <= 120) or (62 <= x < 80 and 102 <= y <= 120):
            return 7
        elif 0 <= x < 80 and 50 <= y < 81:
            return 8
        else:
            return None

    def plot_xg_by_zone(self):
        # Set up figure
        fig, ax = plt.subplots(figsize=(10, 6))
        # Create a boxplot of statsbomb xg by zone
        self.shots_df.boxplot(column='shot_statsbomb_xg', by='zone', ax=ax)
        ax.set_title('Boxplot of StatsBomb xG by Zone')
        ax.set_xlabel('Zone')
        ax.set_ylabel('xG')
        plt.suptitle('')
        plt.show()

    def calculate_zone_proportions(self):
        # Calculate the number of shots taken and the number of goals grouped by zone
        zone_prop = self.shots_df.groupby('zone')['goal'].agg(['count', 'sum'])
        # Calculate the proportion of goals by zone
        zone_prop['proportion'] = zone_prop['sum'] / zone_prop['count']
        return zone_prop

    def split_data(self):
        # Split data into random shuffled train and test sets
        train_df, test_df = train_test_split(self.shots_df, test_size=0.2, random_state=42)
        return train_df, test_df

    def calculate_expected_goals(self, train_df, test_df):
        # Define function to calculate expected goals based on zone proportions
        zone_prop = train_df.groupby('zone')['goal'].agg(['count', 'sum'])
        # Calculate the proportion of shots that resulted in a goal per zone
        zone_prop['proportion'] = zone_prop['sum'] / zone_prop['count']
        # Merge the training sets 'xG' proportion for each test set's shot's zone
        test_df = test_df.merge(zone_prop['proportion'], how='left', left_on='zone', right_index=True)
        # If there is null values, it is due to a divide by zero error.
        # In this case, there will be 0 shots in the zone, and we will say these areas have an xG of 0.
        test_df['expected_goal'] = test_df['proportion'].fillna(0)
        return test_df

    def cross_validate(self, train_df, n_splits=5):
        # Perform 5-fold cross-validation on the training set
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        cv_auc_scores = []
        cv_brier_scores = []
        cv_log_loss_scores = []

        for train_index, val_index in kf.split(train_df):
            # Split the dataset into train and test sets
            cv_train, cv_val = train_df.iloc[train_index], train_df.iloc[val_index]
            # Calculate the xG on the test set
            cv_val = self.calculate_expected_goals(cv_train, cv_val)
            # Get Brier score, ROC AUC and log loss on the xG predictions vs the binary outcomes
            brier_score = brier_score_loss(cv_val['goal'], cv_val['expected_goal'])
            roc_auc = roc_auc_score(cv_val['goal'], cv_val['expected_goal'])
            log_loss_score = log_loss(cv_val['goal'], cv_val['expected_goal'])
            # Append the result of the fold to the scores list for the respective metric
            cv_auc_scores.append(roc_auc)
            cv_brier_scores.append(brier_score)
            cv_log_loss_scores.append(log_loss_score)

        # Print the scores and the overall mean CV score for each metric
        print(f'Cross-validation ROC AUC scores: {cv_auc_scores}')
        print(f'Mean ROC AUC score: {np.mean(cv_auc_scores)}')
        print(f'Cross-validation Brier scores: {cv_brier_scores}')
        print(f'Mean Brier score: {np.mean(cv_brier_scores)}')
        print(f'Cross-validation Log Loss scores: {cv_log_loss_scores}')
        print(f'Mean Log Loss score: {np.mean(cv_log_loss_scores)}')

    def visualize_final_xg_zones(self, xg_values):
        # Function to map xG to color (on red to green)
        def xg_to_color(xg):
            return plt.cm.RdYlGn(xg * 2)  # Multiplied by 2 as largest value was <0.5 and so we wanted highest value to appear green

        # Create the pitch with similar specifications
        pitch = VerticalPitch(pitch_color='#aabb97', line_color='white',
                              stripe_color='#c2d59d', stripe=True, pitch_type='statsbomb', axis=True, label=True, half=True)
        # Set up the pitch figure
        fig, ax = pitch.draw(figsize=(8, 12))

        # Draw zones with xG color mapping
        self.draw_xg_zones(ax, xg_values, xg_to_color)

        # Adding labels for each zone with estimated xG from dataset
        self.add_xg_labels(ax, xg_values)

        plt.show()

    def draw_xg_zones(self, ax, xg_values, xg_to_color):
        # Zone 1
        ax.add_patch(plt.Rectangle((36, 114), 8, 6, fill=True, edgecolor='black', linewidth=2, facecolor=xg_to_color(xg_values[1])))
        # Zone 2
        ax.add_patch(plt.Rectangle((30, 114), 6, 6, fill=True, edgecolor='black', linewidth=2, facecolor=xg_to_color(xg_values[2])))
        ax.add_patch(plt.Rectangle((44, 114), 6, 6, fill=True, edgecolor='black', linewidth=2, facecolor=xg_to_color(xg_values[2])))
        # Zone 3
        ax.add_patch(plt.Rectangle((36, 102), 8, 12, fill=True, edgecolor='black', linewidth=2, facecolor=xg_to_color(xg_values[3])))
        # Zone 4
        ax.add_patch(plt.Rectangle((18, 114), 12, 6, fill=True, edgecolor='black', linewidth=2, facecolor=xg_to_color(xg_values[4])))
        ax.add_patch(plt.Rectangle((50, 114), 12, 6, fill=True, edgecolor='black', linewidth=2, facecolor=xg_to_color(xg_values[4])))
        # Zone 5
        ax.add_patch(plt.Rectangle((18, 102), 18, 12, fill=True, edgecolor='black', linewidth=2, facecolor=xg_to_color(xg_values[5])))
        ax.add_patch(plt.Rectangle((44, 102), 18, 12, fill=True, edgecolor='black', linewidth=2, facecolor=xg_to_color(xg_values[5])))
        # Zone 6
        ax.add_patch(plt.Rectangle((0, 81), 80, 21, fill=True, edgecolor='black', linewidth=2, facecolor=xg_to_color(xg_values[6])))
        # Zone 7
        ax.add_patch(plt.Rectangle((0, 102), 18, 18, fill=True, edgecolor='black', linewidth=2, facecolor=xg_to_color(xg_values[7])))
        ax.add_patch(plt.Rectangle((62, 102), 18, 18, fill=True, edgecolor='black', linewidth=2, facecolor=xg_to_color(xg_values[7])))
        # Zone 8
        ax.add_patch(plt.Rectangle((0, 50), 80, 31, fill=True, edgecolor='black', linewidth=2, facecolor=xg_to_color(xg_values[8])))

    def add_xg_labels(self, ax, xg_values):
        # Adding labels for each zone with estimated xG from dataset
        ax.text(40, 117, 'Zone 1\nxG: {:.3f}'.format(xg_values[1]), ha='center', va='center', fontsize=7, color='black')
        ax.text(33, 117, 'Zone 2\nxG: {:.3f}'.format(xg_values[2]), ha='center', va='center', fontsize=6, color='black')
        ax.text(47, 117, 'Zone 2\nxG: {:.3f}'.format(xg_values[2]), ha='center', va='center', fontsize=6, color='black')
        ax.text(40, 108, 'Zone 3\nxG: {:.3f}'.format(xg_values[3]), ha='center', va='center', fontsize=8, color='black')
        ax.text(24, 117, 'Zone 4\nxG: {:.3f}'.format(xg_values[4]), ha='center', va='center', fontsize=8, color='black')
        ax.text(56, 117, 'Zone 4\nxG: {:.3f}'.format(xg_values[4]), ha='center', va='center', fontsize=8, color='black')
        ax.text(27, 108, 'Zone 5\nxG: {:.3f}'.format(xg_values[5]), ha='center', va='center', fontsize=8, color='black')
        ax.text(53, 108, 'Zone 5\nxG: {:.3f}'.format(xg_values[5]), ha='center', va='center', fontsize=8, color='black')
        ax.text(40, 90, 'Zone 6\nxG: {:.3f}'.format(xg_values[6]), ha='center', va='center', fontsize=8, color='black')
        ax.text(9, 111, 'Zone 7\nxG: {:.3f}'.format(xg_values[7]), ha='center', va='center', fontsize=8, color='black')
        ax.text(71, 111, 'Zone 7\nxG: {:.3f}'.format(xg_values[7]), ha='center', va='center', fontsize=8, color='black')
        ax.text(40, 70, 'Zone 8\nxG: {:.3f}'.format(xg_values[8]), ha='center', va='center', fontsize=8, color='black')

    def run(self):
        # Run the full analysis
        self.load_data()
        self.preprocess_data()
        self.visualize_zones()
        self.determine_zones()
        self.plot_xg_by_zone()
        train_df, test_df = self.split_data()
        self.cross_validate(train_df)

        # Manually input the xG values generated for the method from the entire dataset into a dictionary
        xg_values = {
            1: 0.386,
            2: 0.193,
            3: 0.164,
            4: 0.050,
            5: 0.103,
            6: 0.032,
            7: 0.000,
            8: 0.000
        }

        self.visualize_final_xg_zones(xg_values)

if __name__ == "__main__":
    analysis = RathkeAnalysis()
    analysis.run()

