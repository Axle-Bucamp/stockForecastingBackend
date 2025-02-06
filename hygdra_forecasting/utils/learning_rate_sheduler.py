from enum import Enum
import math


# transformer en class avec lamdba function get next
class cosine_sheduler_mode(str, Enum):
    """
    Enumération définissant les modes du scheduler de taux d'apprentissage.
    
    Attributs:
        COS (str): Mode de décroissance cosinus.
    """
    COS = "cos"

class CosineWarmup():
    """
    Implémente un scheduler de taux d'apprentissage avec une phase de warm-up et une décroissance cosinus.
    
    Attributs:
        learning_rate (float): Taux d'apprentissage initial.
        epochs (int): Nombre total d'époques.
        mode (CosineSchedulerMode): Mode du scheduler (ex: CosineSchedulerMode.COS).
        lr_start (float): Taux d'apprentissage de départ.
        lr_max (float): Taux d'apprentissage maximal après warm-up.
        lr_min (float): Taux d'apprentissage minimal après décroissance.
    """
    def __init__(self, learning_rate:float=0.02, epochs:int=100, mode:cosine_sheduler_mode=cosine_sheduler_mode('cos')):
        """
        Initialise le scheduler de taux d'apprentissage.

        Args:
            learning_rate (float, optionnel): Taux d'apprentissage initial (par défaut 0.02).
            epochs (int, optionnel): Nombre total d'époques (par défaut 100).
            mode (CosineSchedulerMode, optionnel): Mode de scheduler, par défaut CosineSchedulerMode.COS.
        """
        self.learning_rate = learning_rate
        self.mode = mode
        self.lr_start, self.lr_max, self.lr_min = learning_rate, learning_rate * 1.3, learning_rate * 0.3
        self.epochs = epochs
    
    def lrfn(self, epoch):
        """
        Calcule le taux d'apprentissage pour une époque donnée.

        Args:
            epoch (int): L'époque actuelle.

        Returns:
            float: Le taux d'apprentissage ajusté pour l'époque spécifiée.
        """

        # return ....
        lr_ramp_ep = 3 # int(0.02 * epochs)  # 30% of epochs for warm-up
        lr_sus_ep = max(0, int(0.3 * self.epochs) - lr_ramp_ep)
        if epoch < lr_ramp_ep:  # Warm-up phase
            lr = (self.lr_max - self.lr_start) / lr_ramp_ep * epoch + self.lr_start
        elif epoch < lr_ramp_ep + lr_sus_ep:  # Sustain phase at max learning rate
            lr = self.lr_max
        elif self.mode == 'cos':
            decay_total_epochs, decay_epoch_index = self.epochs - lr_ramp_ep - lr_sus_ep, epoch - lr_ramp_ep - lr_sus_ep
            phase = math.pi * decay_epoch_index / decay_total_epochs
            lr = (self.lr_max - self.lr_min) * 0.5 * (1 + math.cos(phase)) + self.lr_min
        else:
            lr = self.lr_min  # Default to minimum learning rate if mode is not recognized

        return lr
    
if __name__ == '__main__':
    scheduler = CosineWarmup()
    
    print("Testing Learning Rate Scheduler:")
    for epoch in [1, 30, 100]:
        print(f"Epoch {epoch}: Learning Rate = {scheduler.lrfn(epoch):.6f}")