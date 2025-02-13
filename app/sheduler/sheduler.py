import time
import schedule
import finetune
import inference

# Scheduler setup
scheduler = schedule.Scheduler()
intervals = ['minutes', 'thirty', 'hour', 'days']

def run_scheduled_tasks(interval):
    tuner = finetune.StockFineTuner(interval=interval, base_weight=f"weight/{interval}/best_model.pth")
    predictor = inference.StockPredictor(interval=interval)
    
    print(f"Starting fine-tuning for interval: {interval}")
    tuner.finetune_many()
    print(f"Completed fine-tuning for interval: {interval}")
    
    print(f"Starting prediction for interval: {interval}")
    predictor.predict()
    print(f"Completed prediction for interval: {interval}")
    
    time.sleep(1)  # Ensure resource cleanup

scheduler.every(1).minutes.do(run_scheduled_tasks, interval='minutes')
scheduler.every(30).minutes.do(run_scheduled_tasks, interval='thirty')
scheduler.every(1).hours.do(run_scheduled_tasks, interval='hour')
scheduler.every(1).days.do(run_scheduled_tasks, interval='days')

if __name__ == "__main__":
    while True:
        scheduler.run_pending()
        time.sleep(1)# Scheduler setup