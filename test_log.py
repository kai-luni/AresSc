from stable_baselines.results_plotter import load_results, ts2xy

log_dir = "log/"


x, y = ts2xy(load_results(log_dir), 'monitor')

print(x)
