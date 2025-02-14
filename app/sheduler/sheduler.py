import time
import schedule
import finetune
import inference

# Scheduler setup
scheduler = schedule.Scheduler()
intervals = ['minutes', 'thirty', 'hour', 'days']

def run_scheduled_tasks_inference(interval):
    predictor = inference.StockPredictor(interval=interval)
     
    print(f"Starting prediction for interval: {interval}")
    predictor.predict()
    print(f"Completed prediction for interval: {interval}")
    
    time.sleep(1)  # Ensure resource cleanup

def run_scheduled_tasks_finetune(interval):
    tuner = finetune.StockFineTuner(interval=interval, base_weight=f"weight/{interval}/best_model.pth")
    print(f"Starting fine-tuning for interval: {interval}")
    tuner.finetune_many()
    print(f"Completed fine-tuning for interval: {interval}")

scheduler.every(1).minutes.do(run_scheduled_tasks_inference, interval='minutes')
scheduler.every(30).minutes.do(run_scheduled_tasks_inference, interval='thirty')
scheduler.every(1).hours.do(run_scheduled_tasks_inference, interval='hour')
scheduler.every(1).days.do(run_scheduled_tasks_inference, interval='days')

"""
scheduler.every(7).days.do(run_scheduled_tasks_finetune, interval='minutes')
scheduler.every(7).days.do(run_scheduled_tasks_finetune, interval='thirty')
scheduler.every(7).days.do(run_scheduled_tasks_finetune, interval='hours')
scheduler.every(7).days.do(run_scheduled_tasks_finetune, interval='minutes')
"""

if __name__ == "__main__":
    while True:
        scheduler.run_pending()
        time.sleep(1)# Scheduler setup