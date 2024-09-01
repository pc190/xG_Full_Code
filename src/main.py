#Import modules
from data_loader import DataLoader
from fcnn import FCNNModel
from fairchild import FairchildModel
from rathke import RathkeAnalysis

def main():
    # Load and preprocess data
    data_loader = DataLoader()
    data_loader.run()

    # Uncomment the model you want to run
    # Run FCNN Model
    model = FCNNModel()
    model.run()

    # Run Fairchild Model
    # model = FairchildModel()
    # model.run()

    # Run Rathke Analysis
    # analysis = RathkeAnalysis()
    # analysis.run()

if __name__ == "__main__":
    main()
