# KesslerGame449

## Setup Instructions

1. Clone the repository to your local machine.
2. Create a virtual environment:
   ```bash
   python -m venv .venv
   ```
3. Activate the virtual environment:
   - On Windows:
     ```bash
     .venv\Scripts\activate
     ```
   - On macOS/Linux:
     ```bash
     source .venv/bin/activate
     ```
4. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```
5. Modify `scenario_test.py` to use the controller of your choice.
   ```python
   # Change this line to use the controller of your choice
   # I'm not sure why it takes two args. Should look into.
   score, perf_data = game.run(scenario=my_test_scenario, controllers=[
                           TestController(), ScottDickController()])
   ```
6. Run `scenario_test.py`
   ```bash
   python scenario_test.py
   ```
