import time
import schedule
import finetune
import inference

# save model, send api call reload model ? (future external server)
scheduler1 = schedule.Scheduler()
scheduler1.every(1).days.do(finetune.finetune_many)

# scheduler predict to csv and overight
# api open and return csv
scheduler2 = schedule.Scheduler()
scheduler2.every(1).days.do(inference.predict_daily)

if __name__ == "__main__" :
    while True:
        scheduler1.run_pending()
        time.sleep(1)
        scheduler2.run_pending()
        time.sleep(1)
