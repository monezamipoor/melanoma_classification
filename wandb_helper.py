import time
import wandb

def wandb_train_log(epoch, loss):
    loss = float(loss)
    wandb.log({"epoch": epoch, "loss": loss})

def wandb_val_log(avg_loss, val_loss, balanced_val_loss, val_metrics, balanced_val_metrics):
    wandb.log({
        "train_loss": avg_loss,
        "val_loss": val_loss,
        "val_balanced_loss": balanced_val_loss,
        **{f"val/{k}": v for k, v in val_metrics.items()},
        **{f"val_balanced/{k}": v for k, v in balanced_val_metrics.items()}
    })

def wandb_test_log(metrics, tag="natural"):
    wandb.log({
        **{f"test_{tag}/{k}": v for k, v in metrics.items()}
    })

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
        print("Wandb disabled")
        wandb.init(mode="disabled")            # prevent logging

    return returnbool

def wandb_watch(model, criterion, log_freq=10):
    wandb.watch(model, criterion, log_freq=log_freq)

def wandb_log_roc(preds, target):
    print(f"Shape of preds: {len(preds)}")
    print(f"Shape of target: {len(target)}")
    print(f"Predictions: {preds[:10]}")
    print(f"Targets: {target[:10]}")

    roc = wandb.plot.roc_curve(
        y_true=target, y_pred=preds, labels=['malignant', 'benign']
    )
    wandb.log({"roc": roc})

# TODO This seems to log and overwrite CMs in wandb and needs changing to log individual matricies
def wandb_log_cm(preds, target, labels, title):
    cm = wandb.plot.confusion_matrix(
        y_true=target, preds=preds, class_names=labels, title=title
    )
    wandb.log({"conf_mat": cm})
