from pathlib import Path
import os
import time
import json
class TrainingLogger:
    def __init__(self, log_dir):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.train_log_file = self.log_dir / "train_metrics.jsonl"
        self.eval_log_file = self.log_dir / "eval_metrics.jsonl"
        with open(self.train_log_file, "w") as f: pass
        with open(self.eval_log_file, "w") as f: pass
    
    def log_train(self, step, epoch, total_loss, policy_loss, kl_div, reward):
        data = {
            "step": step, "epoch": epoch, 
            "total_loss": float(total_loss),
            "policy_loss": float(policy_loss), 
            "kl_div": float(kl_div),
            "reward": float(reward), 
            "timestamp": time.time()
        }
        with open(self.train_log_file, "a") as f:
            f.write(json.dumps(data) + "\n")
            f.flush()
            os.fsync(f.fileno())
            
    def log_eval(self, step, accuracy, mean_reward):
        data = {"step": step, "accuracy": float(accuracy), "mean_reward": float(mean_reward), "timestamp": time.time()}
        with open(self.eval_log_file, "a") as f:
            f.write(json.dumps(data) + "\n")
            f.flush()
            os.fsync(f.fileno())