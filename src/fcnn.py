# Import necessary modules
import pandas as pd
import numpy as np
import ast
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import KFold
from sklearn.metrics import brier_score_loss, roc_auc_score, log_loss
from sklearn.preprocessing import OneHotEncoder
import os
from keras_tuner import HyperModel, RandomSearch

# Define the FCNNHyperModel class inheriting from HyperModel
class FCNNHyperModel(HyperModel):
    def build(self, hp):
        model = Sequential()
        model.add(Flatten(input_shape=(10, 10)))

        for i in range(hp.Int('num_layers', 1, 3)):
            model.add(Dense(units=hp.Int(f'units_{i}', min_value=128, max_value=512, step=128),
                            activation='relu',
                            kernel_regularizer=tf.keras.regularizers.l2(0.01)))
            model.add(BatchNormalization())
            model.add(Dropout(rate=hp.Float(f'dropout_{i}', min_value=0.2, max_value=0.5, step=0.1)))

        model.add(Dense(2, activation='sigmoid'))

        model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=hp.Float('learning_rate', min_value=1e-2, max_value=1e-1, sampling='LOG')),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

        return model

class FCNNModel:
    def __init__(self, base_dir=None, hp_tuning=False):
        # Set seeds for reproducibility
        np.random.seed(42)
        tf.keras.utils.set_random_seed(42)
        # Set the base directory to the project's root
        self.base_dir = base_dir or os.path.dirname(os.getcwd())
        self.trainA = None
        self.df_shot = None
        self.df_freezeframe = None
        self.df_players = None
        self.train_x = None
        self.train_y = None
        self.shot_ids = None
        self.hp_tuning = hp_tuning  # Flag to enable or disable hyperparameter tuning

    def load_data(self, filename='shots_df.csv'):
        # Read in the shots dataframe
        file_path = os.path.join(self.base_dir, 'data', filename)
        self.trainA = pd.read_csv(file_path)

    def preprocess_data(self):
        # Exclude penalties from the dataset
        self.trainA = self.trainA[self.trainA['shot_type'] != 'Penalty']

        # Fill null freeze frames with an empty string and convert to list of dictionaries
        self.trainA['freeze_frame'] = self.trainA['freeze_frame'].fillna('[]')
        self.trainA['freeze_frame'] = self.trainA['freeze_frame'].apply(ast.literal_eval)

        # Filter out rows without a non-teammate goalkeeper
        self.trainA = self.trainA[self.trainA['freeze_frame'].apply(self.has_non_teammate_goalkeeper)]

    @staticmethod
    def has_non_teammate_goalkeeper(freeze_frame):
        # Check if there's a goalkeeper who is not a teammate in the freeze frame
        for player in freeze_frame:
            if player.get('keeper', False) and not player.get('teammate', False):
                return True
        return False

    def select_columns(self):
        # Select relevant columns for the neural network model
        self.df_shot = self.trainA[['id', 'minute', 'match_id', 'period', 'play_pattern',
                                    'position', 'possession_team', 'shot_body_part', 'shot_first_time',
                                    'shot_outcome', 'shot_statsbomb_xg', 'shot_technique', 'shot_type',
                                    'home_team', 'away_team', 'competition', 'goal']]

        # Save freeze frame data separately
        self.df_freezeframe = self.trainA[['id', 'freeze_frame']]

        # Convert freeze_frame back to a list of dictionaries if necessary
        self.df_freezeframe['freeze_frame'] = self.df_freezeframe['freeze_frame'].apply(lambda x: ast.literal_eval(str(x)) if isinstance(x, str) else x)

        # Apply function to create shot_type_engineered column
        self.df_shot['shot_type_engineered'] = self.df_shot.apply(self.categorize_shot, axis=1)

    @staticmethod
    def categorize_shot(row):
        # Create a new column combining shot_body_part and shot_technique into categories
        if row['shot_technique'] == 'Normal' and row['shot_body_part'] in ['Right Foot', 'Left Foot']:
            return 'Foot Normal'
        elif row['shot_technique'] in ['Half Volley', 'Volley', 'Lob'] and row['shot_body_part'] in ['Right Foot', 'Left Foot']:
            return 'Foot Volley'
        elif row['shot_body_part'] == 'Head':
            return 'Head'
        else:
            return 'Other'  # Includes backheel and overhead kick

    def process_freeze_frame(self):
        # Process freeze frame data into a new DataFrame for player information
        new_rows = []

        # For each player in the freeze frame
        for index, row in self.df_freezeframe.iterrows():
            shot_id = row['id']
            freeze_frame = row['freeze_frame']

            # If the freeze frame is iterable
            if isinstance(freeze_frame, list):
                # For each player in the freeze frame
                for player in freeze_frame:
                    x, y = player.get('location', [None, None])
                    new_row = {
                        'shot_id': shot_id,
                        'actor': player.get('actor', None),
                        'teammate': player.get('teammate', None),
                        'keeper': player.get('keeper', None),
                        'x': x,
                        'y': y
                    }
                    new_rows.append(new_row)
            else:
                print(f"Row {index} has non-iterable freeze_frame: {freeze_frame}")

        # Create players dataframe from new_rows
        self.df_players = pd.DataFrame(new_rows)
        self.df_players = self.df_players[(self.df_players['actor'] == True) | (self.df_players['teammate'] == False)]
        self.df_players.loc[self.df_players['keeper'] == True, 'teammate'] = True

    @staticmethod
    def calculate_distance(x1, y1, x2, y2):
        # Function to calculate Euclidean distance between 2 points
        return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    @staticmethod
    def calculate_angle(x1, y1, x2, y2):
        # Function to calculate angle between the line spanning the shooter to the center of the goal and the defender
        goal_x, goal_y = 120, 40
        vec_goal_x, vec_goal_y = goal_x - x1, goal_y - y1
        vec_player_x, vec_player_y = x2 - x1, y2 - y1
        dot_product = vec_goal_x * vec_player_x + vec_goal_y * vec_player_y
        magnitude_goal = np.sqrt(vec_goal_x ** 2 + vec_goal_y ** 2)
        magnitude_player = np.sqrt(vec_player_x ** 2 + vec_player_y ** 2)
        cos_angle = dot_product / (magnitude_goal * magnitude_player)
        angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
        return np.degrees(angle)

    @staticmethod
    def calculate_angle_to_goal(x, y):
        # Function to calculate the angle to the goal (angle of goal available)
        A = np.array([120, 36])
        B = np.array([120, 44])
        C = np.array([x, y])
        CA = A - C
        CB = B - C
        dot_product = np.dot(CA, CB)
        magnitude_CA = np.linalg.norm(CA)
        magnitude_CB = np.linalg.norm(CB)
        cosine_angle = dot_product / (magnitude_CA * magnitude_CB)
        cosine_angle = np.clip(cosine_angle, -1, 1)
        angle_radians = np.arccos(cosine_angle)
        return np.degrees(angle_radians)

    def process_tracking_data(self):
        # Process the positional data of each player
        shooters = self.df_players[self.df_players['actor']].copy()
        shooters.set_index('shot_id', inplace=True, drop=True)
        shotId_shooter_map = shooters[['x', 'y']].to_dict(orient='index')
        self.df_players['shooter_x'] = self.df_players['shot_id'].apply(lambda val: shotId_shooter_map[val]['x'])
        self.df_players['shooter_y'] = self.df_players['shot_id'].apply(lambda val: shotId_shooter_map[val]['y'])

        self.df_players['distance_to_shooter'] = self.df_players.apply(
            lambda row: self.calculate_distance(row['shooter_x'], row['shooter_y'], row['x'], row['y']), axis=1)
        self.df_players['angle_to_shooter'] = self.df_players.apply(
            lambda row: self.calculate_angle(row['shooter_x'], row['shooter_y'], row['x'], row['y']), axis=1)
        self.df_players['distance_to_goal'] = self.df_players.apply(
            lambda row: self.calculate_distance(row['x'], row['y'], 120, 40), axis=1)
        self.df_players['angle_to_goal'] = self.df_players.apply(
            lambda row: self.calculate_angle_to_goal(row['x'], row['y']), axis=1)

    def prepare_training_data(self):
        # Sort by shot_id and reset index for joining consistency
        self.df_players.sort_values(by=['shot_id'], inplace=True)
        self.df_players.reset_index(drop=True, inplace=True)

        # Sort by shot_id and reset index for joining consistency
        self.df_shot.sort_values(by=['id'], inplace=True)
        self.df_shot.reset_index(drop=True, inplace=True)

        # Group the players by shot id
        grouped = self.df_players.groupby('shot_id')
        grouped = sorted(grouped, key=lambda x: x[0])

        # Initialize training data arrays
        self.train_x = np.zeros([len(grouped), 10, 6])

        # Initialize i
        i = 0
        # Save shot_ids for later results comparison
        self.shot_ids = self.df_shot.id.values

        # For each shot
        for name, group in grouped:
            # If final shot, end loop
            if i >= len(self.shot_ids):
                print("Index out of bounds for shot_ids. Exiting loop.")
                break

            # Debugging if joining mismatch
            if name != self.shot_ids[i]:
                print(f"Error: shot_id mismatch at index {i} (expected {self.shot_ids[i]}, got {name})")
                continue

            # Define shooter, defenders, and gk id's for the shot
            shooter_ids = group[group.actor].index
            defense_ids = group[~group.teammate & ~group.keeper].index
            gk_ids = group[group.keeper].index

            # If the goalkeeper exists
            if len(gk_ids) > 0:
                # Get goalkeeper distance and angle to shooter
                [gk_dist, gk_angle] = group.loc[gk_ids, ['distance_to_shooter', 'angle_to_shooter']].values[0]
            else:
                print("No goalkeeper in picture")
                gk_dist, gk_angle = 0, 0  # Default values if no goalkeeper

            # Get the shooter distance and angle from the goal
            [shooter_dist, shooter_angle] = group.loc[shooter_ids, ['distance_to_goal', 'angle_to_goal']].values[0]

            # For each defender (maximum 10)
            for j in range(10):
                if j < len(defense_ids):
                    # Get defender distance and angle to shooter
                    defense_id = defense_ids[j]
                    [defender_dist, defender_angle] = group.loc[defense_id, ['distance_to_shooter', 'angle_to_shooter']].values
                else:
                    # Defender is assumed to be very far away behind the shooter
                    defender_dist, defender_angle = 50, 180

                # Save to array
                self.train_x[i, j, :] = [shooter_dist, shooter_angle, defender_dist, defender_angle, gk_dist, gk_angle]

            # Proceed to next iteration
            i += 1

        # Reshape train_x to 2D
        num_samples, num_timesteps, num_features = self.train_x.shape
        train_x_reshaped = self.train_x.reshape(num_samples * num_timesteps, num_features)

        # Calculate mean and standard deviation for each feature
        means = train_x_reshaped.mean(axis=0)
        stds = train_x_reshaped.std(axis=0)

        # Standardize the features
        train_x_reshaped = (train_x_reshaped - means) / stds

        # Reshape back to 3D
        self.train_x = train_x_reshaped.reshape(num_samples, num_timesteps, num_features)

        # Prepare the data. y includes goals only
        self.train_x = np.nan_to_num(self.train_x)
        self.train_y = self.df_shot[['goal']].to_numpy()

        # One-hot encode target labels
        self.train_y = to_categorical(self.train_y, num_classes=2)

        # One-hot encode categorical training features
        categorical_features = ['shot_type_engineered']
        encoder = OneHotEncoder(sparse_output=False)
        encoded_features = encoder.fit_transform(self.df_shot[categorical_features])

        # Expand dimensions of the encoded features to match train_x
        encoded_features_expanded = np.repeat(encoded_features[:, np.newaxis, :], 10, axis=1)

        # Concatenate the one-hot encoded features to train_x
        self.train_x = np.concatenate([self.train_x, encoded_features_expanded], axis=2)

    def build_model(self, hp=None):
        if self.hp_tuning and hp is not None:
            # Build model with hyperparameters provided by Keras Tuner
            hypermodel = FCNNHyperModel()
            return hypermodel.build(hp)
        else:
            # Build the Fully Connected Neural Network (FCNN) Model
            model = Sequential()
            # Flatten the input (10, 10) shape
            model.add(Flatten(input_shape=(10, 10)))

            model.add(Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)))
            model.add(BatchNormalization())
            model.add(Dropout(0.2))

            model.add(Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)))
            model.add(BatchNormalization())
            model.add(Dropout(0.2))

            model.add(Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)))
            model.add(BatchNormalization())
            model.add(Dropout(0.2))

            model.add(Dense(2, activation='sigmoid'))

            # Compile the model
            model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.1), loss='binary_crossentropy', metrics=['accuracy'])

            return model

    def tune_model(self):
        # Set up Keras Tuner for hyperparameter tuning
        tuner = RandomSearch(
            FCNNHyperModel(),  # Use FCNNHyperModel instead of FCNNModel
            objective='val_loss',
            max_trials=20,
            executions_per_trial=5,
            directory='tuner_results',
            project_name='fcnn_tuning'
        )

        # Search for the best hyperparameters
        tuner.search(
            self.train_x, self.train_y,
            epochs=200,
            batch_size=32,
            validation_split=0.2,
            callbacks=[EarlyStopping(monitor='val_loss', patience=10)]
        )

        # Retrieve the best hyperparameters
        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
        print(f"Best hyperparameters: {best_hps.values}")
        return best_hps

    def cross_validate_model(self):
        if self.hp_tuning:
            # Perform hyperparameter tuning if enabled
            best_hps = self.tune_model()
        else:
            best_hps = None

        # Initialize KFold cross-validator
        kf = KFold(n_splits=5, shuffle=True, random_state=42)

        # Initialize lists to store scores and predictions
        brier_scores, auc_scores, log_losses, all_predictions = [], [], [], []

        # Cross-validation process
        for train_index, test_index in kf.split(self.train_x):
            # Split the data
            X_train, X_test = self.train_x[train_index], self.train_x[test_index]
            y_train, y_test = self.train_y[train_index], self.train_y[test_index]

            # Build the FCNN model
            model = self.build_model(best_hps)

            # Callbacks including early stopping and reducing learning rate on plateau
            early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
            reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, min_lr=0.00001)

            # Train the model
            model.fit(X_train, y_train, epochs=200, batch_size=32, validation_split=0.2, verbose=1,
                      callbacks=[reduce_lr, early_stopping])

            # Predict probabilities on the test set
            pred_probabilities = model.predict(X_test)

            # Convert y_test back from one-hot encoding
            actual_labels = np.argmax(y_test, axis=1)

            # Extract the prediction probabilities for the positive class (assuming class 1 is the positive class)
            xg_cnn = pred_probabilities[:, 1]

            # Collect shot ids for the current fold
            test_shot_ids = self.shot_ids[test_index]

            # Collect predictions and true values for the current fold
            for shot_id, statsbomb_xg, pred_prob, goal in zip(test_shot_ids, self.df_shot.loc[test_index, 'shot_statsbomb_xg'], xg_cnn, actual_labels):
                all_predictions.append({'shot_id': shot_id, 'statsbomb_xg': statsbomb_xg, 'prediction': pred_prob, 'goal': goal})

            # Get Brier score, ROC AUC, and log loss on the xG predictions vs the binary outcomes
            brier = brier_score_loss(actual_labels, xg_cnn)
            auc = roc_auc_score(actual_labels, xg_cnn)
            logloss = log_loss(actual_labels, pred_probabilities)

            # Append the result of the fold to the scores list for the respective metric
            brier_scores.append(brier)
            auc_scores.append(auc)
            log_losses.append(logloss)

        # Print the scores and the overall mean CV score for each metric
        print(f'Cross-validation Brier scores: {brier_scores}')
        print(f'Mean Brier score: {np.mean(brier_scores)}')
        print(f'Cross-validation AUC scores: {auc_scores}')
        print(f'Mean AUC score: {np.mean(auc_scores)}')
        print(f'Cross-validation Log Loss scores: {log_losses}')
        print(f'Mean Log Loss: {np.mean(log_losses)}')

        return all_predictions

    def save_predictions(self, predictions_df):
        # Save predictions to CSV for use in results_compare.py
        output_path = os.path.join(self.base_dir, 'results', 'cnn_results.csv')
        predictions_df.to_csv(output_path, index=False)

    def run(self):
        # Run the full analysis pipeline
        self.load_data()
        self.preprocess_data()
        self.select_columns()
        self.process_freeze_frame()
        self.process_tracking_data()
        self.prepare_training_data()

        # Perform cross-validation and get predictions
        all_predictions = self.cross_validate_model()

        # Create predictions DataFrame
        predictions_df = pd.DataFrame(all_predictions)

        # Calculate deviance residuals column for comparison
        predictions_df['diff'] = abs(predictions_df['statsbomb_xg'] - predictions_df['prediction'])
        print(predictions_df.head())

        # Save the predictions
        self.save_predictions(predictions_df)

if __name__ == "__main__":
    # Instantiate and run the model with or without hyperparameter tuning
    model = FCNNModel(hp_tuning=False)  # Set to True to enable tuning. Results will be saved to tuner_results folder
    model.run()
    
    
    
    
    