import os
import torch

def main():
    # Check CUDA availability
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"CUDA is available. Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("CUDA not available. Using CPU.")
    
    print("Nature Inspired Computation - Final Project")
    print("1. Run Phase 1 (Model Optimization & Training)")
    print("2. Run Phase 2 (Meta-Optimization & XAI)")
    
    choice = input("Select phase (1/2): ")
    
    if choice == '1':
        os.system("python run_phase1.py")
    elif choice == '2':
        if not os.path.exists("results/best_model.pth"):
            print("Error: Phase 1 must be run first to generate the model!")
        else:
            os.system("python run_phase2.py")
    else:
        print("Invalid choice")

if __name__ == "__main__":
    main()