import time
import wandb

def wandb_train_log(epoch, loss):
    loss = float(loss)
    wandb.log({"epoch": epoch, "loss": loss})

def wandb_val_log(avg_loss, val_loss, **val_metrics):
    wandb.log({"train_loss": avg_loss, "val_loss": val_loss, **val_metrics})

def wandb_login(opt):
    returnbool = False

    runid = (opt['model']['backbone']
             + '-' + opt['training']['optimizer']
             + '-' + str(opt['dataset']['batch_size'])
             + '-' + str(opt['training']['learning_rate'])
             + '-' + time.strftime("%Y%m%d-%H%M%S"))
    try:
        if wandb.login(key=opt['testing']['wandb']['api_key'], relogin=True):  # Check we have a valid login
            wandb.init(project=opt['testing']['wandb']['project_name'],
                       entity=opt['testing']['wandb']['entity'],
                       id=runid)
            wandb.config.update(opt)
            wandb.define_metric("*", step_metric="epoch")
            returnbool = True  # Successful configuration
    except Exception as e:
        print("Failed to configure wandb")
        wandb.init(mode="disabled")            # prevent logging

    return returnbool

def wandb_watch(model, criterion, log_freq=10):
    wandb.watch(model, criterion, log_freq=log_freq)
