from header import *
import torch
class DeepSpeedAgent:

    def __init__(self, model, args):
        super(DeepSpeedAgent, self).__init__()
        self.args = args
        self.model = model
        self.load_stage_1_parameters(args["delta_ckpt_path"])
        self.load_feature_clusters(args.get("feature_cluster_path", None))


        for name, param in self.model.named_parameters():
            param.requires_grad = False

        for name, param in self.model.image_decoder.named_parameters():
            param.requires_grad = True

        for name, param in self.model.prompt_learner.named_parameters():
            param.requires_grad = True




        # load config parameters of deepspeed
        ds_params = json.load(open(self.args['ds_config_path']))
        ds_params['scheduler']['params']['total_num_steps'] = self.args['total_steps']
        ds_params['scheduler']['params']['warmup_num_steps'] = max(10, int(self.args['total_steps'] * self.args['warmup_rate']))
        self.ds_engine, self.optimizer, _ , _ = deepspeed.initialize(
            model=self.model,
            model_parameters=self.model.parameters(),
            config_params=ds_params,
            dist_init_required=True,
            args=types.SimpleNamespace(**args)
        )

    @torch.no_grad()
    def predict(self, batch):
        self.model.eval()
        string = self.model.generate_one_sample(batch)
        return string

    def train_model(self, batch, current_step=0, pbar=None):
        self.ds_engine.module.train()
        loss, mle_acc = self.ds_engine(batch)

        self.ds_engine.backward(loss)
        self.ds_engine.step()
        pbar.set_description(f'[!] loss: {round(loss.item(), 4)}; token_acc: {round(mle_acc*100, 2)}')
        pbar.update(1)
        if self.args['local_rank'] == 0 and self.args['log_path'] and current_step % self.args['logging_step'] == 0:
            elapsed = pbar.format_dict['elapsed']
            rate = pbar.format_dict['rate']
            remaining = (pbar.total - pbar.n) / rate if rate and pbar.total else 0
            remaining = str(datetime.timedelta(seconds=remaining))
            logging.info(f'[!] progress: {round(pbar.n/pbar.total, 5)}; remaining time: {remaining}; loss: {round(loss.item(), 4)}; token_acc: {round(mle_acc*100, 2)}')

        mle_acc *= 100
        return mle_acc

    def save_model(self, path, current_step):
        # only save trainable model parameters
        param_grad_dic = {
            k: v.requires_grad for (k, v) in self.ds_engine.module.named_parameters()
        }
        state_dict = self.ds_engine.module.state_dict()
        checkpoint = OrderedDict()
        for k, v in self.ds_engine.module.named_parameters():
            if v.requires_grad:
                print(k)
                # checkpoint[k] = v.to(device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
                checkpoint[k] = v.to(torch.device("cpu"))
        torch.save(checkpoint, f'{path}/pytorch_model.pt')
        # save tokenizer
        self.model.llama_tokenizer.save_pretrained(path)
        # save configuration
        self.model.llama_model.config.save_pretrained(path)
        print(f'[!] save model into {path}')
        # **保存 FeatureCluster 聚类中心**
        self.model.feature_cluster.save_centroids(f"{path}/feature_clusters.pth")

        print(f'[!] 模型已保存至 {path}')

    def load_stage_1_parameters(self, path):
        delta_ckpt = torch.load(path, map_location=torch.device('cpu'))
        self.model.load_state_dict(delta_ckpt, strict=False)

    def load_feature_clusters(self, path=None):
        """
        加载自适应聚类中心。如果文件存在则加载，否则在训练过程中重新计算。
        如果参数未传入则使用默认的 self.model.feature_cluster.save_path 作为路径。
        """
        if path is None:
            path = self.model.feature_cluster.save_path

        if os.path.exists(path):
            self.model.feature_cluster.load_centroids()
            print(f"[INFO] Loaded feature cluster centers from: {path}")
        else:
            print(f"[WARNING] Feature cluster center file not found at: {path}. Will recalculate during training.")
