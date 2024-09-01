import pandas as pd
import ast
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss
import numpy as np
import math
import statsmodels.api as sm
import os

class FairchildModel:
    def __init__(self, base_dir=None):
        # Set the base directory to the project's root
        self.base_dir = base_dir or os.path.dirname(os.getcwd())
        self.shots_df = None
        self.fc_df = None

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

        # Get distance of shot from byline and calculate shot angle
        self.shots_df['shot_distance_fc'] = 120 - self.shots_df['x_shot']
        self.shots_df['shot_angle_fc'] = self.shots_df.apply(self.calculate_shot_angle_fc, axis=1)

    @staticmethod
    def has_non_teammate_goalkeeper(freeze_frame):
        # For every player in the freezeframe
        for player in freeze_frame:
            # Return true if there is a player tagged as a goalkeeper and not a teammate
            if player.get('keeper', False) and not player.get('teammate', False):
                return True
        return False

    @staticmethod
    def calculate_shot_angle_fc(row):
        # Function that calculates shot angle

        # Define fixed points A and B representing the goalposts
        A = np.array([120, 36])
        B = np.array([120, 44])

        # Define point C, representing the position of the shot
        C = np.array([row['x_shot'], row['y_shot']])

        # Create vectors CA and CB
        CA = A - C
        CB = B - C

        # Calculate the dot product of CA and CB
        dot_product = np.dot(CA, CB)

        # Calculate the magnitudes of CA and CB
        magnitude_CA = np.linalg.norm(CA)
        magnitude_CB = np.linalg.norm(CB)

        # Calculate the cosine of the angle
        cosine_angle = dot_product / (magnitude_CA * magnitude_CB)

        # Ensure the cosine value is within the valid range [-1, 1]
        cosine_angle = np.clip(cosine_angle, -1, 1)

        # Calculate the angle in radians
        angle_radians = np.arccos(cosine_angle)

        # Convert the angle to degrees. This represents our shot angle
        angle_degrees = np.degrees(angle_radians)

        return angle_degrees

    def create_fc_dataframe(self):
        # Create simplified dataframe using only variables considered by Fairchild
        self.fc_df = self.shots_df[['id', 'goal', 'shot_body_part', 'shot_first_time', 'shot_freeze_frame',
                                    'shot_statsbomb_xg', 'shot_technique', 'shot_type', 'under_pressure',
                                    'freeze_frame', 'x_shot', 'y_shot', 'shot_distance_fc', 'shot_angle_fc']]

        # Group free kicks and corners into a single category called set piece
        self.fc_df['shot_type'] = self.fc_df['shot_type'].apply(lambda x: 'Set Piece' if x in ['Free Kick', 'Corner'] else x)

    def calculate_defender_features(self):
        # Apply the function to create the new feature for the number of defenders in sight
        self.fc_df['num_defs'] = self.fc_df.apply(lambda row: self.count_defenders_in_sight(row['freeze_frame'],
                                                                                            row['x_shot'],
                                                                                            row['y_shot']), axis=1)

        # Apply the function to create a new column 'closest_defender_distance'
        self.fc_df['closest_defender_distance'] = self.fc_df.apply(lambda row: self.calculate_closest_defender_distance(
            row['freeze_frame'], row['x_shot'], row['y_shot']), axis=1)

    @staticmethod
    def count_defenders_in_sight(shot_freeze_frame, shot_location_x, shot_location_y):
        # Function to count the number of defenders in the ball goal triangle

        # Define goalposts positions
        goal_post_left = (120, 36)
        goal_post_right = (120, 44)

        # Initialize 0 defenders in sight
        defenders_in_sight = 0

        # For each player in the freeze frame
        for player in shot_freeze_frame:
            # If the player is a non-teammate and not a goalkeeper
            if not player['teammate'] and not player['keeper']:
                # Get player location
                px, py = player['location']
                # If the player location is within the ball-goal triangle
                if FairchildModel.is_point_in_triangle(px, py, shot_location_x, shot_location_y,
                                                       goal_post_left[0], goal_post_left[1],
                                                       goal_post_right[0], goal_post_right[1]):
                    # Add 1 to defenders_in_sight
                    defenders_in_sight += 1

        return defenders_in_sight

    @staticmethod
    def is_point_in_triangle(px, py, ax, ay, bx, by, cx, cy):
        # Using barycentric coordinates to check if point (px, py) is in triangle (ax, ay), (bx, by), (cx, cy)
        v0x, v0y = cx - ax, cy - ay
        v1x, v1y = bx - ax, by - ay
        v2x, v2y = px - ax, py - ay

        dot00 = v0x * v0x + v0y * v0y
        dot01 = v0x * v1x + v0y * v1y
        dot02 = v0x * v2x + v0y * v2y
        dot11 = v1x * v1x + v1y * v1y
        dot12 = v1x * v2x + v1y * v2y

        invDenom = 1 / (dot00 * dot11 - dot01 * dot01)
        u = (dot11 * dot02 - dot01 * dot12) * invDenom
        v = (dot00 * dot12 - dot01 * dot02) * invDenom

        return (u >= 0) and (v >= 0) and (u + v < 1)

    @staticmethod
    def calculate_closest_defender_distance(shot_freeze_frame, shot_location_x, shot_location_y):
        # Function to calculate the distance of the closest defender

        # Initialize the minimum distance with a large number i.e.infinity
        min_distance = float('inf')

        # Iterate through each player in the freeze frame
        for player in shot_freeze_frame:
            # Check if the player is a defender (not a teammate and not the keeper)
            if not player['teammate'] and not player['keeper']:
                # Define defender location
                defender_location = player['location']
                # Calculate the Euclidean distance between the shot location and the defender location
                distance = math.sqrt((defender_location[0] - shot_location_x)**2 + (defender_location[1] - shot_location_y)**2)
                # Update the minimum distance if the current distance is smaller
                if distance < min_distance:
                    min_distance = distance

        # Return the minimum distance
        return min_distance

    def categorize_shot_type(self):
        # Create a new column combining shot_body_part and shot_technique into categories
        self.fc_df['shot_type_engineered'] = self.fc_df.apply(self.categorize_shot, axis=1)

    @staticmethod
    def categorize_shot(row):
        # Function to categorize the shot type
        if row['shot_technique'] == 'Normal' and row['shot_body_part'] in ['Right Foot', 'Left Foot']:
            return 'Foot Normal'
        elif row['shot_technique'] in ['Half Volley', 'Volley', 'Lob'] and row['shot_body_part'] in ['Right Foot', 'Left Foot']:
            return 'Foot Volley'
        # Volleys and half volleys were seen to perform identically
        elif row['shot_body_part'] == 'Head':
            return 'Head'
        else:
            return 'Other'  # Includes backheel and overhead kick

    def prepare_model_data(self):
        # Create a dataframe including only the final variables included in the model. Include also statsbombxg for later comparison
        fc_model_df = self.fc_df[['id', 'shot_distance_fc', 'shot_angle_fc', 'shot_type_engineered',
                                  'num_defs', 'closest_defender_distance', 'goal', 'shot_statsbomb_xg']]

        # Rename id as shot_id for consistency
        fc_model_df.rename(columns={'id': 'shot_id'}, inplace=True)

        # Encode categorical variables
        fc_model_df = pd.get_dummies(fc_model_df, columns=['shot_type_engineered'], drop_first=True)

        return fc_model_df

    def train_model(self, fc_model_df):
        # Define the features and target variable (goal)
        X = fc_model_df.drop(['goal', 'shot_id', 'shot_statsbomb_xg'], axis=1)
        y = fc_model_df['goal']

        # Add a constant to ensure the model is unbiased
        X = sm.add_constant(X)

        # Split the entire data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Set up and train the logistic regression model
        model = sm.Logit(y_train.astype(float), X_train.astype(float))
        result = model.fit()

        return result, X, y, X_train, y_train

    def cross_validate_model(self, X, y, fc_model_df, shot_ids):
        # Perform 5-fold cross-validation on the training set
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        cv_auc_scores = []
        cv_brier_scores = []
        cv_log_loss_scores = []
        all_predictions = []

        # For each of the 5 folds
        for train_index, val_index in kf.split(X):
            # Split the dataset into train and test sets
            cv_X_train, cv_X_val = X.iloc[train_index], X.iloc[val_index]
            cv_y_train, cv_y_val = y.iloc[train_index], y.iloc[val_index]

            # Set up the model using statsmodels
            model_cv = sm.Logit(cv_y_train.astype(float), cv_X_train.astype(float))
            # Fit the model
            result_cv = model_cv.fit(disp=0)

            # Calculate the probability of each test point being in the positive class to represent xG
            cv_y_pred_proba = result_cv.predict(cv_X_val.astype(float))

            # Get Brier score, ROC AUC, and log loss on the xG predictions vs the binary outcomes
            roc_auc = roc_auc_score(cv_y_val, cv_y_pred_proba)
            brier_score = brier_score_loss(cv_y_val, cv_y_pred_proba)
            logloss = log_loss(cv_y_val, cv_y_pred_proba)
            # Append the result of the fold to the scores list for the respective metric
            cv_auc_scores.append(roc_auc)
            cv_brier_scores.append(brier_score)
            cv_log_loss_scores.append(logloss)

            # Collect shot ids for the current fold
            test_ids = fc_model_df.index[val_index]
            test_shot_ids = shot_ids[val_index]

            # Collect predictions and true values for the current fold
            for shot_id, statsbomb_xg, pred_prob, goal in zip(test_shot_ids, fc_model_df.loc[test_ids, 'shot_statsbomb_xg'], cv_y_pred_proba, cv_y_val):
                all_predictions.append({
                    'shot_id': shot_id,
                    'statsbomb_xg': statsbomb_xg,
                    'prediction': pred_prob,
                    'goal': goal
                })

        # Print the scores and the overall mean CV score for each metric
        print(f'Cross-validation ROC AUC scores: {cv_auc_scores}')
        print(f'Mean ROC AUC score: {np.mean(cv_auc_scores)}')
        print(f'Cross-validation Brier scores: {cv_brier_scores}')
        print(f'Mean Brier score: {np.mean(cv_brier_scores)}')
        print(f'Cross-validation Log Loss scores: {cv_log_loss_scores}')
        print(f'Mean Log Loss score: {np.mean(cv_log_loss_scores)}')

        return all_predictions

    def save_predictions(self, predictions_df):
        # Save to CSV for use in results_compare.py
        output_path = os.path.join(self.base_dir, 'results', 'fc_results.csv')
        predictions_df.to_csv(output_path, index=False)

    def run(self):
        # Run the full analysis
        self.load_data()
        self.preprocess_data()
        self.create_fc_dataframe()
        self.calculate_defender_features()
        self.categorize_shot_type()

        # Prepare the model data
        fc_model_df = self.prepare_model_data()

        # Train the model
        result, X, y, X_train, y_train = self.train_model(fc_model_df)

        # Perform cross-validation and get predictions
        shot_ids = fc_model_df['shot_id'].values
        all_predictions = self.cross_validate_model(X, y, fc_model_df, shot_ids)

        # Create predictions DataFrame
        predictions_df = pd.DataFrame(all_predictions)

        # Calculate deviance residuals column for comparison
        predictions_df['diff'] = abs(predictions_df['statsbomb_xg'] - predictions_df['prediction'])
        print(predictions_df.head())

        # Save the predictions
        self.save_predictions(predictions_df)


if __name__ == "__main__":
    model = FairchildModel()
    model.run()
