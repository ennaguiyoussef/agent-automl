# main.py
import pandas as pd
import numpy as np
from pathlib import Path
from langgraph.types import Command

from agent_automl.graph import app

def generate_realistic_dataset():
    np.random.seed(42)
    n = 1000
    age = np.random.randint(18, 70, n)
    income = np.random.randint(30000, 120000, n)
    contract_type = np.random.choice(['monthly', 'yearly', 'two_year'], n)
    monthly_charges = np.random.uniform(20.0, 120.0, n)
    
    churn_prob = (monthly_charges / 120.0) + (np.where(contract_type == 'monthly', 0.4, 0.0))
    churn = (churn_prob + np.random.normal(0, 0.1, n) > 0.6).astype(int)
    
    df = pd.DataFrame({
        'customer_id': range(1, n + 1),
        'age': age, 'income': income, 'contract_type': contract_type,
        'monthly_charges': np.round(monthly_charges, 2), 'churn': churn
    })
    df.loc[np.random.choice(n, 50), 'income'] = np.nan
    return df

def main():
    dataset_path = "realistic_churn_data.csv"
    generate_realistic_dataset().to_csv(dataset_path, index=False)
    print(f"Generated dataset with 1000 rows.\n")

    initial_state = {
        "user_query": "Predict whether a customer will churn.",
        "dataset_path": dataset_path, 
        "dataset_profile": None,
        "plan": None,
        "evaluation_report": None,
        "messages": []
    }

    # Fresh thread ID to avoid checkpoint corruption
    config = {"configurable": {"thread_id": "final-clean-session"}} 

    print("="*60)
    print("STARTING AGENTIC AUTOML SYSTEM")
    print("="*60)
    
    # 1. Run the graph. It will AUTOMATICALLY PAUSE before execute_ml.
    print("-> Running initial analysis...")
    final_state = app.invoke(initial_state, config=config)
    
    plan = final_state.get('plan')
    if not plan:
        print("Error: Graph finished without a valid plan.")
        return

    # 2. HUMAN-IN-THE-LOOP INTERACTION
    print("\n" + "="*60)
    print("AGENT PROPOSAL (GRAPH PAUSED)")
    print("="*60)
    print(f"Problem Type: {plan.problem_type}")
    print(f"Target Column: {plan.target_column}")
    print(f"Selected Models: {', '.join(plan.selected_models)}")
    print(f"Reasoning: {plan.reasoning}")
    
    print("\n" + "-"*60)
    print("COMMANDS:")
    print("  'approve'                  -> Train the models.")
    print("  'change target <name>'     -> Change the target column and re-plan.")
    print("  'reject'                   -> Abort the process.")
    print("-"*60)
    
    user_input = input("\nEnter your command: ").strip().lower()

    if user_input == "approve":
        print("\n-> Resuming graph to execute ML pipeline...")
        # Resume execution. It will proceed to execute_ml.
        final_state = app.invoke(Command(resume=True), config=config)

    elif user_input.startswith("change target"):
        new_target = user_input.replace("change target", "").strip()
        print(f"\n-> Updating context to target '{new_target}' and asking Agent to re-plan...")
        
        new_query = f"Predict the value of '{new_target}'. (Note: User corrected target to '{new_target}'). Re-evaluate problem type and models."
        
        # Atomically update state and jump back to planner
        final_state = app.invoke(
            Command(
                update={"plan": None, "user_query": new_query},
                goto="analyze_and_plan"
            ),
            config=config
        )

    elif user_input == "reject":
        print("\n-> Process aborted by user. No models trained.")
        return
    else:
        print("\n-> Invalid command. Aborting.")
        return

    # 3. Print Final Results
    report = final_state.get('evaluation_report')
    if report and report.best_model:
        print("\n" + "="*60)
        print("FINAL ML RESULTS")
        print("="*60)
        print(f"Best Model: {report.best_model} (Score: {report.best_metric:.4f})")
        for res in report.results:
            print(f" - {res.model_name}: {res.metrics}")
    else:
        print("\n[Info] ML Execution was skipped or aborted.")

if __name__ == "__main__":
    main()