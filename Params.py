import DataHandeling
import os
from datetime import datetime

ROOT_SAVE_DIR = os.path.join('.', 'Logs')
ROOT_DATA_DIR = os.path.join('.', 'Data')


class Params(object):
    # Net Parameters
    net_build_params = {'layers': 2, 'kernel_size': 7, 'kernel_num': 32, 'num_repeat': 3}

    # Hardware
    use_gpu = True  # IF NO GPU AVAILABE, SET TO FALSE
    gpu_id = 0  # IF MORE THAN 1 GPU IS AVAILABLE, SELECT WHICH ONE
    dry_run = False  # SET TO TRUE IF YOU DO NOT WANT TO SAVE ANY OUTPUTS (GOOD WHEN DEBUGGING)
    profile = False  # SET TO TRUE FOR THROUGHPUT PROFILING

    batch_size = 50  # number of images per batch
    num_iterations = 1000000  # total number of iterations
    learning_rate = 1e-3  # learning rate

    crops_per_image = 5  # number of random crops per image, used for faster training
    crop_size = (16, 16)  # crop size of the input image

    # Training Regime

    # Weight for loss on Foreground, Background and Edge
    # class_weights = [0.04, 0.06, 0.9]
    class_weights = [0.14, 0.16, 0.7]

    # Validation
    validation_interval = 100  # number of train iterations between each validation step

    # Loading Checkpoints
    # Load model weights from checkpoint
    load_checkpoint = False
    # Path to checkpoint
    # Best
    # load_checkpoint_path = os.path.join(ROOT_SAVE_DIR, 'fusion_2GPU', '2018-07-20_105331',
    #                                     'model_605000.ckpt')
    load_checkpoint_path = os.path.join(ROOT_SAVE_DIR, 'fusion_2GPU', '2018-07-23_180639',
                                        'model_967000.ckpt')
    # Saving Checkpoints
    experiment_name = 'fusion_2GPU'  # Name of your experiment
    save_checkpoint_dir = ROOT_SAVE_DIR  # Path to save files. set at top of this file
    save_checkpoint_iteration = 500  # number of iteration between each checkpoint save
    save_checkpoint_every_N_hours = 5
    save_checkpoint_max_to_keep = 5

    # Tensorboard
    write_to_tb_interval = 100  # # number of iteration between each print to tensorboard

    """
    DO NOT TOUCH OR CHANGE THIS CLASS UNLESS YOU KNOW WHAT YOU ARE DOING
    """

    def __init__(self):  # DO NOT TOUCH THIS METHOD
        self.train_q_capacity = 50 * self.batch_size  # DO NOT TOUCH
        self.val_q_capacity = 20 * self.batch_size  # DO NOT TOUCH
        self.min_after_dequeue = 10 * self.batch_size  # DO NOT TOUCH
        self.data_provider_class = DataHandeling.CSVSegReaderRandom  # DO NOT TOUCH

        self.norm = 2 ** 7  # NORMALIZATION OF THE INPUT IMAGE. PLEASE DO NOT TOUCH

        self.data_base_folder = [ROOT_DATA_DIR]  # DO NOT TOUCH
        self.val_data_base_folder = [ROOT_DATA_DIR]  # DO NOT TOUCH

        self.train_csv_file = [os.path.join(db, 'train.csv') for db in self.data_base_folder]  # DO NOT TOUCH
        self.val_csv_file = [os.path.join(db, 'val.csv') for db in self.val_data_base_folder]  # DO NOT TOUCH
        self.save_log_dir = ROOT_SAVE_DIR

        # Data and Data Provider
        self.root_data_dir = ROOT_DATA_DIR  # THE DIRECTORY OF THE DATA, SET THIS AT THE TOP OF THE FILE

        self.train_data_provider = self.data_provider_class(self.train_csv_file, image_size=(512, 640),
                                                            crop_size=self.crop_size,
                                                            crops_per_image=self.crops_per_image,
                                                            num_threads=4, capacity=self.train_q_capacity,
                                                            min_after_dequeue=self.min_after_dequeue,
                                                            num_examples=None, data_format='NHWC',
                                                            is_random=True)

        self.val_data_provider = self.data_provider_class(self.val_csv_file, image_size=(512, 640),
                                                          crop_size=self.crop_size,
                                                          crops_per_image=self.crops_per_image,
                                                          num_threads=4, capacity=self.val_q_capacity,
                                                          min_after_dequeue=self.min_after_dequeue,
                                                          num_examples=None, data_format='NHWC',
                                                          is_random=True)
        os.environ['CUDA_VISIBLE_DEVICES'] = str(self.gpu_id)
        now_string = datetime.now().strftime('%Y-%m-%d_%H%M%S')
        self.experiment_log_dir = os.path.join(self.save_log_dir, self.experiment_name, now_string)
        self.experiment_save_dir = os.path.join(self.save_checkpoint_dir, self.experiment_name,
                                                now_string)
        if not self.dry_run:
            os.makedirs(self.experiment_log_dir, exist_ok=True)
            os.makedirs(self.experiment_save_dir, exist_ok=True)
            os.makedirs(os.path.join(self.experiment_log_dir, 'train'), exist_ok=True)
            os.makedirs(os.path.join(self.experiment_log_dir, 'val'), exist_ok=True)
