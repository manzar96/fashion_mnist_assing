

class Config:

    def __init__(self,args,device):
        self.device = device
        if args.modelckpt is not None:
            self.checkpoint_dir = args.modelckpt
        else:
            self.checkpoint_dir = './checkpoints/draft/'
        self.logdir = 'events/'+ "".join(self.checkpoint_dir.split(
            "checkpoints/")[1:])
        if args.loadckpt is not None:
            self.loadckpt = args.loadckpt
        else:
            self.loadckpt = None
        self.clip = args.clip
        self.skip_val = args.skip_validation
        self.patience = args.patience
        self.early_stopping = not args.not_use_early_stopping

        self.batch_size = args.bs
        self.epochs = args.es
        self.lr = args.lr

