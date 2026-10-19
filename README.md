# RL-Based Active Boring Bar Vibration Control

A Python simulation project that studies whether a Soft Actor-Critic (SAC) reinforcement-learning controller can reduce vibration in a simplified boring-bar model. The project compares **No control**, **PID**, and **SAC**, evaluates performance across multiple sinusoidal disturbance conditions, and provides a Streamlit interface backed by a FastAPI service.

> **Scope and limitations:** This is a simulation study, not a physical experiment or validated digital twin. The plant is a simplified single-mode mass–spring–damper model driven by a sinusoidal disturbance. The SAC action is mapped directly to a bounded control force; magnetic-actuator current-to-force conversion and actuator dynamics are not explicitly modelled. The simulated displacement is a modal coordinate and should not be interpreted as calibrated physical tool-tip displacement. The model does not represent regenerative chatter, and the current implementation does not include a Kalman filter or noisy sensor model.

## How it works

The simulated plant follows

$$
 m\ddot q + c\dot q + kq = F_{disturbance}(t) + F_{control}(t).
$$

The SAC agent receives normalized displacement, velocity, and the previous action, then selects a bounded control action. The simulation applies that action as control force. Its performance is compared with no control and a PID baseline using steady-state RMS vibration and control-effort metrics.

The nominal plant parameters are informed by a published boring-bar vibration study, but the overall plant and control setup remain simplified and are not a reproduction of the paper's complete actuator model.

## Repository structure

```text
config.py                 # Plant, disturbance, training and evaluation settings
models.py                 # Single-mode plant and sinusoidal disturbance
controllers.py            # No-control and PID controller implementations
env.py                    # Gymnasium environment used to train SAC
sac_training.py           # SAC model construction and training
experiments.py            # Standard comparisons, metrics and plots
run_baseline.py           # Train SAC and run standard comparisons
tune_pid.py                # Tune PID gains on designated tuning conditions
robust_evaluation.py      # Evaluate controllers over an 84-condition grid
deployment_service.py     # Shared inference/service logic
api.py                    # FastAPI endpoints
app.py                    # Streamlit interface
requirements.txt          # Python dependencies
results/                  # Saved policies, metrics, time series and figures
```

## 1. Fork and clone the repository

1. Open the GitHub repository page.
2. Click **Fork** and create a fork under your GitHub account.
3. Open your fork, click **Code**, copy the HTTPS URL, and run the following in Windows Command Prompt (CMD). Replace the URL with your fork's URL and choose your preferred local folder.

```bat
git clone https://github.com/YOUR-USERNAME/YOUR-REPOSITORY.git
cd YOUR-REPOSITORY
```

If you have downloaded the source as a ZIP instead, extract it and open CMD in the extracted project root. The project root is the folder containing `config.py`, `app.py`, `api.py`, and `requirements.txt`.

## 2. Create the Python environment

Use Python 3.10 or 3.11 if possible. From the project root, run:

```bat
py -3.11 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If the `py -3.11` launcher is unavailable but `python` points to a supported Python version, use `python -m venv .venv` instead. Whenever you open a new CMD window, activate the environment again with `.venv\Scripts\activate`.

## 3. Generate results in sequence

Run these commands from the project root, with `.venv` activated. The commands write files under `results/`; back up any existing results you want to preserve before rerunning them.

### Step 1 — Train SAC and run the standard comparison

The project uses a 300,000-timestep SAC run for the current main training configuration. Save this policy as the versioned model so the evaluation and deployment steps can use the same file:

```bat
python run_baseline.py --timesteps 300000 --seed 42 --model-path results/sac_vibration_model_v2
```

This trains SAC and then runs the standard experiments. It creates/updates the model archive `results/sac_vibration_model_v2.zip` and the standard comparison CSVs and plots in `results/`.

For a quick code/dependency smoke test, you can use fewer timesteps (for example, `--timesteps 10000`). That is **not** the full training run and its results should not be treated as the main reported results.

To rerun only the standard experiments using an already-trained policy, pass the saved model path and skip training:

```bat
python run_baseline.py --skip-training --model-path results/sac_vibration_model_v2
```

### Step 2 — Tune and validate the PID baseline

```bat
python tune_pid.py
```

This writes the tuning summary and selected gains to `results/pid_tuning_results.csv` and `results/tuned_pid.json`, and evaluates the selected gains on separate validation conditions in `results/pid_validation_conditions.csv`. Keep the validation conditions separate from tuning when interpreting results.

### Step 3 — Run the 84-condition robustness evaluation

```bat
python robust_evaluation.py
```

This compares No control, PID, and SAC across 84 combinations of disturbance amplitude, frequency, and phase. It writes:

- `results/robust_evaluation_by_condition.csv`
- `results/robust_evaluation_summary.csv`
- `results/robust_evaluation_rms.png`
- `results/robust_evaluation_reduction.png`

**Run Steps 1–3 in order** when generating a fresh set of results. The robust evaluation expects `results/sac_vibration_model_v2.zip` and `results/tuned_pid.json` to exist. If you retrain SAC, rerun the evaluations so the CSVs and figures correspond to the new policy.

## 4. Launch the FastAPI service

The API serves simulations from the saved policy; it does not train SAC during inference. Start it from the project root in one CMD window:

```bat
.venv\Scripts\python.exe -m uvicorn api:app --reload
```

The API is normally available at `http://127.0.0.1:8000`. Open `http://127.0.0.1:8000/docs` to inspect and try the interactive API documentation. The service exposes health/model-status routes and simulation/comparison endpoints. Only **No control**, **PID**, and **SAC** are supported as deployment controller choices; the offline evaluation plots may include additional research comparisons, but those are not part of the Streamlit/API controller list.

The deployment service has a default model path. Before launching, verify that `deployment_service.py` uses `results/sac_vibration_model_v2.zip` as `DEFAULT_MODEL_PATH` if you intend to serve the v2 policy. Alternatively, configure `SAC_MODEL_PATH` to point to the model archive. Restart the API process after changing the path.

## 5. Launch the Streamlit interface

Open a **second** CMD window, navigate to the project root, and activate the environment:

```bat
.venv\Scripts\activate
python -m streamlit run app.py
```

Streamlit normally opens at `http://localhost:8501`. Keep the FastAPI server running in the first terminal. The sidebar should show that the API is connected and the SAC model is available. The interface should offer **No control**, **PID**, and **SAC**; it does not expose the offline-only Active damping comparison.

To stop either server, focus its CMD window and press `Ctrl+C`.

## Results and interpretation

- `controller_metrics.csv` and `controller_timeseries.csv` contain the standard experiment metrics and time-series data.
- `robust_evaluation_by_condition.csv` contains per-condition results; `robust_evaluation_summary.csv` aggregates them.
- `pid_tuning_results.csv`, `tuned_pid.json`, and `pid_validation_conditions.csv` document PID selection and validation.
- PNG files in `results/` contain standard and robust-evaluation plots.

RMS reduction is calculated relative to the no-control case for the corresponding condition. A positive reduction means less vibration; a negative reduction means the controller increased vibration. Do not confuse the mean of per-condition percentage reductions with the percentage change between aggregate mean RMS values; these are different statistics. Report the metric definition, test conditions, and saved model version alongside any headline result.

Results can change when the policy is retrained, the seed/configuration is changed, or evaluation conditions are modified. The saved CSVs describe a particular run and should not be presented as evidence of real-machine performance.

## Reproducibility notes

- Use the same Python environment and `requirements.txt` for repeat runs.
- Record the training seed, timestep count, model archive, and evaluation configuration.
- Do not tune PID gains on the final robustness test grid.
- Keep the model archive and result files from the same run together.
- Before publishing results, confirm that the API and Streamlit are loading the intended model archive.

## Reference

Fan Chen and Guangya Liu, “Active damping of machine tool vibrations and cutting force measurement with a magnetic actuator,” *The International Journal of Advanced Manufacturing Technology*, 89, 691–700 (2017). The paper informs the vibration-model context; this code is a simplified simulation and does not implement the paper's complete magnetic-actuator dynamics.+