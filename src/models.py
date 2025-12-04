import torch
import torch.nn as nn
from transformers import GPT2Model, GPT2Config
from tqdm import tqdm
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression, Lasso
import warnings
from sklearn import tree
import xgboost as xgb
from moe import TransformerModel_var_moe
from base_models import NeuralNetwork, ParallelNetworks


def build_model(conf):
    if conf.family == "gpt2":
        model = TransformerModel(
            n_dims=conf.n_dims,
            n_positions=conf.n_positions,
            n_embd=conf.n_embd,
            n_layer=conf.n_layer,
            n_head=conf.n_head,
        )
    elif  conf.family == "gpt2_var_glu":
        model = TransformerModel_var_glu(
            glu_type=conf.glu_type,
            n_dims=conf.n_dims,
            n_positions=conf.n_positions,
            n_embd=conf.n_embd,
            n_layer=conf.n_layer,
            n_head=conf.n_head,
        )
    elif conf.family == "gpt2_var_moe":
        model = TransformerModel_var_moe(
            n_dims=conf.n_dims,
            n_positions=conf.n_positions,
            n_embd=conf.n_embd,
            n_layer=conf.n_layer,
            n_head=conf.n_head,
            num_experts=conf.num_experts, 
        )
        

    else:
        raise NotImplementedError

    return model


def get_relevant_baselines(task_name):
    task_to_baselines = {
        "linear_regression": [
            (LeastSquaresModel, {}),
            (NNModel, {"n_neighbors": 3}),
            (AveragingModel, {}),
        ],
        "linear_classification": [
            (NNModel, {"n_neighbors": 3}),
            (AveragingModel, {}),
        ],
        "sparse_linear_regression": [
            (LeastSquaresModel, {}),
            (NNModel, {"n_neighbors": 3}),
            (AveragingModel, {}),
        ]
        + [(LassoModel, {"alpha": alpha}) for alpha in [1, 0.1, 0.01, 0.001, 0.0001]],
        "relu_2nn_regression": [
            (LeastSquaresModel, {}),
            (NNModel, {"n_neighbors": 3}),
            (AveragingModel, {}),
            (
                GDModel,
                {
                    "model_class": NeuralNetwork,
                    "model_class_args": {
                        "in_size": 20,
                        "hidden_size": 100,
                        "out_size": 1,
                    },
                    "opt_alg": "adam",
                    "batch_size": 100,
                    "lr": 5e-3,
                    "num_steps": 100,
                },
            ),
        ],
        "decision_tree": [
            (LeastSquaresModel, {}),
            (NNModel, {"n_neighbors": 3}),
            (DecisionTreeModel, {"max_depth": 4}),
            (DecisionTreeModel, {"max_depth": None}),
            (XGBoostModel, {}),
            (AveragingModel, {}),
        ],
        "polynomial_regression": [
        (LeastSquaresModel, {}),
        (ChebyshevFitModel, {"max_degree": 11}), #max_degree should change according to the max_degree we train
        (NNModel, {"n_neighbors": 3}),
        (DecisionTreeModel, {"max_depth": 4}),
        ],
    }

    models = [model_cls(**kwargs) for model_cls, kwargs in task_to_baselines[task_name]]
    return models

##############################################
#baseline model for polynomial function
import torch

def chebyshev_polynomials(x, degree):
    """
    x: 1D tensor shape [N] or 2D [B, N]
    Returns: if input 1D -> [N, degree+1]
             if input 2D -> [B, N, degree+1]
    """
    squeeze_input = False
    if x.ndim == 1:
        x = x.unsqueeze(0)  # (1, N)
        squeeze_input = True

    B, N = x.shape
    D = degree + 1
    T = torch.zeros(B, N, D, device=x.device, dtype=x.dtype)
    T[:, :, 0] = 1.0
    if degree >= 1:
        T[:, :, 1] = x
    for n in range(1, degree):
        T[:, :, n + 1] = 2 * x * T[:, :, n] - T[:, :, n - 1]

    if squeeze_input:
        return T.squeeze(0)  # [N, D]
    return T  # [B, N, D]


class ChebyshevFitModel:
    """
    Vectorized ICL-style Chebyshev baseline with adaptive/degenerate degree:
      - zero-shot: constant predictor (mean of available y)
      - few-shot: fit_degree = min(k-1, max_degree) (degenerate)
      - many-shot: fit up to max_degree
    Behavior: for each k (1..N-1) use first k points to predict x_next = x[k],
              store prediction at preds[:, k].
    """
    def __init__(self, max_degree=11, l2_reg=1e-6, device='cpu', dtype=torch.float32):
        self.max_degree = max_degree
        self.l2_reg = l2_reg
        self.name = "ChebyshevFitModel"
        self.device = device
        self.dtype = dtype

    def __call__(self, xs, ys, inds=None):
        # xs: [B, N, 1], ys: [B, N]
        xs, ys = xs.to(self.device).type(self.dtype), ys.to(self.device).type(self.dtype)
        B, N, _ = xs.shape
        Dmax = self.max_degree + 1

        preds = torch.zeros(B, N, device=self.device, dtype=self.dtype)

        if N <= 1:
            # trivial: if only 0 or 1 point, fill with constant (y0 or 0)
            if N == 1:
                preds[:, 0] = ys[:, 0]
            return preds

        # Precompute Chebyshev basis for all x for every item in batch
        # T_all: [B, N, Dmax]
        T_all = chebyshev_polynomials(xs.view(B, N), self.max_degree)  # returns [B, N, Dmax]

        # Precompute per-row outer products t_i @ t_i^T and t_i * y_i
        # t_i: [B, N, Dmax]
        t = T_all  # alias
        # outer per point: [B, N, Dmax, Dmax]
        # careful with memory: for moderate Dmax (<= 12) and N not huge, OK
        outer = t.unsqueeze(3) * t.unsqueeze(2)  # broadcasting -> [B, N, Dmax, Dmax]
        # Xty term per point: [B, N, Dmax]
        Xty_per_point = t * ys.unsqueeze(2)  # [B, N, Dmax]

        # Cumulative sums along points -> prefix sums for k=1..N
        # XtX_prefix[k-1] = sum_{i=0..k-1} outer[:, i]
        XtX_prefix = outer.cumsum(dim=1)  # [B, N, Dmax, Dmax]
        Xty_prefix = Xty_per_point.cumsum(dim=1)  # [B, N, Dmax]

        I_full = torch.eye(Dmax, device=self.device, dtype=self.dtype).unsqueeze(0).expand(B, Dmax, Dmax)

        # zero-shot: fallback (we set preds[:,0]); use mean of available y if want
        # For consistency with earlier choices, set preds[:,0] = mean of ys[:0]? set to 0 or ys[:,0]
        # We'll set preds[:,0] = ys[:,0] if available, else 0
        preds[:, 0] = ys[:, 0]

        # Loop over k (1..N-1): use first k points (prefix index k-1) to predict x_next = x[k]
        # This loop iterates N-1 times (typically small), but inner ops are batched over B.
        for k in range(1, N):
            # determine fit degree: deg = min(k-1, max_degree)
            fit_degree = min(k - 1, self.max_degree)
            d = fit_degree + 1  # number of basis used

            # slice prefix matrices to first d dims
            XtX_k = XtX_prefix[:, k - 1, :d, :d]  # [B, d, d]
            Xty_k = Xty_prefix[:, k - 1, :d]      # [B, d]

            # regularization: add lambda * I_d
            # build I_d with batch
            I_d = torch.eye(d, device=self.device, dtype=self.dtype).unsqueeze(0).expand(B, d, d)
            A = XtX_k + self.l2_reg * I_d  # [B, d, d]
            b = Xty_k.unsqueeze(2)         # [B, d, 1]

            # If d == 1, can solve directly
            # Use Cholesky for robustness and batch solve
            # ensure A is symmetric positive definite (reg helps)
            try:
                L = torch.linalg.cholesky(A)        # [B, d, d]
                c = torch.cholesky_solve(b, L).squeeze(2)  # [B, d]
            except RuntimeError:
                # fallback to torch.linalg.solve if cholesky fails
                # solve for each batch (vectorized solve available)
                c = torch.linalg.solve(A, b).squeeze(2)  # [B, d]

            # Build T_next for x_next: need first d basis entries
            T_next_full = T_all[:, k, :d]  # [B, d]
            # predicted y_next for each batch
            y_pred_k = (T_next_full * c).sum(dim=1)  # [B]
            preds[:, k] = y_pred_k

        return preds
###########################################################################


###########################################################################
import torch
import torch.nn as nn
import torch.nn.functional as F

class GLUMLP(nn.Module):
    """
    GLU-based Feedforward layer replacing GPT2MLP.
    Supports: 'glu', 'geglu', 'swiglu'
    """
    def __init__(self, n_embd, hidden_dim=None, glu_type="geglu"):
        super().__init__()
        hidden_dim = hidden_dim or 4 * n_embd

        self.w1 = nn.Linear(n_embd, hidden_dim)
        self.w2 = nn.Linear(n_embd, hidden_dim)
        self.out = nn.Linear(hidden_dim, n_embd)

        self.glu_type = glu_type.lower()

    def forward(self, x):
        a = self.w1(x)
        b = self.w2(x)

        if self.glu_type == "glu":
            gated = a * torch.sigmoid(b)
        elif self.glu_type == "geglu":
            gated = a * F.gelu(b)
        elif self.glu_type == "swiglu":
            gated = a * F.silu(b)
        else:
            raise ValueError(f"Unknown GLU type {self.glu_type}")

        return self.out(gated)



class CustomGPT2Model(GPT2Model):
    """
    GPT2Model but with GLU-based MLP in every transformer block.
    """
    def __init__(self, config, glu_type="geglu"):
        super().__init__(config)

        n_embd = config.n_embd
        hidden_dim = 4 * n_embd

        # change mlp to GLUMLP
        for block in self.h:
            block.mlp = GLUMLP(
                n_embd=n_embd,
                hidden_dim=hidden_dim,
                glu_type=glu_type,
            )
        self.glu_type = glu_type

#copy the original transformerModel and change the backbone:
class TransformerModel_var_glu(nn.Module):
    def __init__(self, glu_type,n_dims, n_positions, n_embd=128, n_layer=12, n_head=4):
        super(TransformerModel_var_glu, self).__init__()
        configuration = GPT2Config(
            n_positions=2 * n_positions,
            n_embd=n_embd,
            n_layer=n_layer,
            n_head=n_head,
            resid_pdrop=0.0,
            embd_pdrop=0.0,
            attn_pdrop=0.0,
            use_cache=False,
        )
        self.name = f"gpt2_embd={n_embd}_layer={n_layer}_head={n_head}"

        self.n_positions = n_positions
        self.n_dims = n_dims
        self._read_in = nn.Linear(n_dims, n_embd)
        self._backbone = CustomGPT2Model(configuration, glu_type=glu_type) #change the backbone to our customGPT2Model
        self._read_out = nn.Linear(n_embd, 1)

    @staticmethod
    def _combine(xs_b, ys_b):
        """Interleaves the x's and the y's into a single sequence."""
        bsize, points, dim = xs_b.shape
        ys_b_wide = torch.cat(
            (
                ys_b.view(bsize, points, 1),
                torch.zeros(bsize, points, dim - 1, device=ys_b.device),
            ),
            axis=2,
        )
        zs = torch.stack((xs_b, ys_b_wide), dim=2)
        zs = zs.view(bsize, 2 * points, dim)
        return zs

    def forward(self, xs, ys, inds=None):
        if inds is None:
            inds = torch.arange(ys.shape[1])
        else:
            inds = torch.tensor(inds)
            if max(inds) >= ys.shape[1] or min(inds) < 0:
                raise ValueError("inds contain indices where xs and ys are not defined")
        zs = self._combine(xs, ys)
        embeds = self._read_in(zs)
        output = self._backbone(inputs_embeds=embeds).last_hidden_state
        prediction = self._read_out(output)
        return prediction[:, ::2, 0][:, inds]  # predict only on xs
###########################################################################

class TransformerModel(nn.Module):
    def __init__(self,n_dims, n_positions, n_embd=128, n_layer=12, n_head=4):
        super(TransformerModel, self).__init__()
        configuration = GPT2Config(
            n_positions=2 * n_positions,
            vocab_size=0,  
            n_embd=n_embd,
            n_layer=n_layer,
            n_head=n_head,
            resid_pdrop=0.0,
            embd_pdrop=0.0,
            attn_pdrop=0.0,
            use_cache=False,
        )
        self.name = f"gpt2_embd={n_embd}_layer={n_layer}_head={n_head}"

        self.n_positions = n_positions
        self.n_dims = n_dims
        self._read_in = nn.Linear(n_dims, n_embd)
        self._backbone = GPT2Model(configuration)
        self._read_out = nn.Linear(n_embd, 1)

    @staticmethod
    def _combine(xs_b, ys_b):
        """Interleaves the x's and the y's into a single sequence."""
        bsize, points, dim = xs_b.shape
        ys_b_wide = torch.cat(
            (
                ys_b.view(bsize, points, 1),
                torch.zeros(bsize, points, dim - 1, device=ys_b.device),
            ),
            axis=2,
        )
        zs = torch.stack((xs_b, ys_b_wide), dim=2)
        zs = zs.view(bsize, 2 * points, dim)
        return zs

    def forward(self, xs, ys, inds=None):
        if inds is None:
            inds = torch.arange(ys.shape[1])
        else:
            inds = torch.tensor(inds)
            if max(inds) >= ys.shape[1] or min(inds) < 0:
                raise ValueError("inds contain indices where xs and ys are not defined")
        zs = self._combine(xs, ys)
        embeds = self._read_in(zs)
        output = self._backbone(inputs_embeds=embeds).last_hidden_state
        prediction = self._read_out(output)
        return prediction[:, ::2, 0][:, inds]  # predict only on xs


class NNModel:
    def __init__(self, n_neighbors, weights="uniform"):
        # should we be picking k optimally
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.name = f"NN_n={n_neighbors}_{weights}"

    def __call__(self, xs, ys, inds=None):
        if inds is None:
            inds = range(ys.shape[1])
        else:
            if max(inds) >= ys.shape[1] or min(inds) < 0:
                raise ValueError("inds contain indices where xs and ys are not defined")

        preds = []

        for i in inds:
            if i == 0:
                preds.append(torch.zeros_like(ys[:, 0]))  # predict zero for first point
                continue
            train_xs, train_ys = xs[:, :i], ys[:, :i]
            test_x = xs[:, i : i + 1]
            dist = (train_xs - test_x).square().sum(dim=2).sqrt()

            if self.weights == "uniform":
                weights = torch.ones_like(dist)
            else:
                weights = 1.0 / dist
                inf_mask = torch.isinf(weights).float()  # deal with exact match
                inf_row = torch.any(inf_mask, axis=1)
                weights[inf_row] = inf_mask[inf_row]

            pred = []
            k = min(i, self.n_neighbors)
            ranks = dist.argsort()[:, :k]
            for y, w, n in zip(train_ys, weights, ranks):
                y, w = y[n], w[n]
                pred.append((w * y).sum() / w.sum())
            preds.append(torch.stack(pred))

        return torch.stack(preds, dim=1)


# xs and ys should be on cpu for this method. Otherwise the output maybe off in case when train_xs is not full rank due to the implementation of torch.linalg.lstsq.
class LeastSquaresModel:
    def __init__(self, driver=None):
        self.driver = driver
        self.name = f"OLS_driver={driver}"

    def __call__(self, xs, ys, inds=None):
        xs, ys = xs.cpu(), ys.cpu()
        if inds is None:
            inds = range(ys.shape[1])
        else:
            if max(inds) >= ys.shape[1] or min(inds) < 0:
                raise ValueError("inds contain indices where xs and ys are not defined")

        preds = []

        for i in inds:
            if i == 0:
                #
                preds.append(torch.zeros_like(ys[:, 0]))  # predict zero for first point
                continue
            train_xs, train_ys = xs[:, :i], ys[:, :i]
            test_x = xs[:, i : i + 1]

            ws, _, _, _ = torch.linalg.lstsq(
                train_xs, train_ys.unsqueeze(2), driver=self.driver
            )

            pred = test_x @ ws
            preds.append(pred[:, 0, 0])

        return torch.stack(preds, dim=1)


class AveragingModel:
    def __init__(self):
        self.name = "averaging"

    def __call__(self, xs, ys, inds=None):
        if inds is None:
            inds = range(ys.shape[1])
        else:
            if max(inds) >= ys.shape[1] or min(inds) < 0:
                raise ValueError("inds contain indices where xs and ys are not defined")

        preds = []

        for i in inds:
            if i == 0:
                preds.append(torch.zeros_like(ys[:, 0]))  # predict zero for first point
                continue
            train_xs, train_ys = xs[:, :i], ys[:, :i]
            test_x = xs[:, i : i + 1]

            train_zs = train_xs * train_ys.unsqueeze(dim=-1)
            w_p = train_zs.mean(dim=1).unsqueeze(dim=-1)
            pred = test_x @ w_p
            preds.append(pred[:, 0, 0])

        return torch.stack(preds, dim=1)


# Lasso regression (for sparse linear regression).
# Seems to take more time as we decrease alpha.
class LassoModel:
    def __init__(self, alpha, max_iter=100000):
        # the l1 regularizer gets multiplied by alpha.
        self.alpha = alpha
        self.max_iter = max_iter
        self.name = f"lasso_alpha={alpha}_max_iter={max_iter}"

    # inds is a list containing indices where we want the prediction.
    # prediction made at all indices by default.
    def __call__(self, xs, ys, inds=None):
        xs, ys = xs.cpu(), ys.cpu()

        if inds is None:
            inds = range(ys.shape[1])
        else:
            if max(inds) >= ys.shape[1] or min(inds) < 0:
                raise ValueError("inds contain indices where xs and ys are not defined")

        preds = []  # predict one for first point

        # i: loop over num_points
        # j: loop over bsize
        for i in inds:
            pred = torch.zeros_like(ys[:, 0])

            if i > 0:
                pred = torch.zeros_like(ys[:, 0])
                for j in range(ys.shape[0]):
                    train_xs, train_ys = xs[j, :i], ys[j, :i]

                    # If all points till now have the same label, predict that label.

                    clf = Lasso(
                        alpha=self.alpha, fit_intercept=False, max_iter=self.max_iter
                    )

                    # Check for convergence.
                    with warnings.catch_warnings():
                        warnings.filterwarnings("error")
                        try:
                            clf.fit(train_xs, train_ys)
                        except Warning:
                            print(f"lasso convergence warning at i={i}, j={j}.")
                            raise

                    w_pred = torch.from_numpy(clf.coef_).unsqueeze(1)

                    test_x = xs[j, i : i + 1]
                    y_pred = (test_x @ w_pred.float()).squeeze(1)
                    pred[j] = y_pred[0]

            preds.append(pred)

        return torch.stack(preds, dim=1)


# Gradient Descent and variants.
# Example usage: gd_model = GDModel(NeuralNetwork, {'in_size': 50, 'hidden_size':400, 'out_size' :1}, opt_alg = 'adam', batch_size = 100, lr = 5e-3, num_steps = 200)
class GDModel:
    def __init__(
        self,
        model_class,
        model_class_args,
        opt_alg="sgd",
        batch_size=1,
        num_steps=1000,
        lr=1e-3,
        loss_name="squared",
    ):
        # model_class: torch.nn model class
        # model_class_args: a dict containing arguments for model_class
        # opt_alg can be 'sgd' or 'adam'
        # verbose: whether to print the progress or not
        # batch_size: batch size for sgd
        self.model_class = model_class
        self.model_class_args = model_class_args
        self.opt_alg = opt_alg
        self.lr = lr
        self.batch_size = batch_size
        self.num_steps = num_steps
        self.loss_name = loss_name

        self.name = f"gd_model_class={model_class}_model_class_args={model_class_args}_opt_alg={opt_alg}_lr={lr}_batch_size={batch_size}_num_steps={num_steps}_loss_name={loss_name}"

    def __call__(self, xs, ys, inds=None, verbose=False, print_step=100):
        # inds is a list containing indices where we want the prediction.
        # prediction made at all indices by default.
        # xs: bsize X npoints X ndim.
        # ys: bsize X npoints.
        xs, ys = xs.cuda(), ys.cuda()

        if inds is None:
            inds = range(ys.shape[1])
        else:
            if max(inds) >= ys.shape[1] or min(inds) < 0:
                raise ValueError("inds contain indices where xs and ys are not defined")

        preds = []  # predict one for first point

        # i: loop over num_points
        for i in tqdm(inds):
            pred = torch.zeros_like(ys[:, 0])
            model = ParallelNetworks(
                ys.shape[0], self.model_class, **self.model_class_args
            )
            model.cuda()
            if i > 0:
                pred = torch.zeros_like(ys[:, 0])

                train_xs, train_ys = xs[:, :i], ys[:, :i]
                test_xs, test_ys = xs[:, i : i + 1], ys[:, i : i + 1]

                if self.opt_alg == "sgd":
                    optimizer = torch.optim.SGD(model.parameters(), lr=self.lr)
                elif self.opt_alg == "adam":
                    optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
                else:
                    raise NotImplementedError(f"{self.opt_alg} not implemented.")

                if self.loss_name == "squared":
                    loss_criterion = nn.MSELoss()
                else:
                    raise NotImplementedError(f"{self.loss_name} not implemented.")

                # Training loop
                for j in range(self.num_steps):

                    # Prepare batch
                    mask = torch.zeros(i).bool()
                    perm = torch.randperm(i)
                    mask[perm[: self.batch_size]] = True
                    train_xs_cur, train_ys_cur = train_xs[:, mask, :], train_ys[:, mask]

                    if verbose and j % print_step == 0:
                        model.eval()
                        with torch.no_grad():
                            outputs = model(train_xs_cur)
                            loss = loss_criterion(
                                outputs[:, :, 0], train_ys_cur
                            ).detach()
                            outputs_test = model(test_xs)
                            test_loss = loss_criterion(
                                outputs_test[:, :, 0], test_ys
                            ).detach()
                            print(
                                f"ind:{i},step:{j}, train_loss:{loss.item()}, test_loss:{test_loss.item()}"
                            )

                    optimizer.zero_grad()

                    model.train()
                    outputs = model(train_xs_cur)
                    loss = loss_criterion(outputs[:, :, 0], train_ys_cur)
                    loss.backward()
                    optimizer.step()

                model.eval()
                pred = model(test_xs).detach()

                assert pred.shape[1] == 1 and pred.shape[2] == 1
                pred = pred[:, 0, 0]

            preds.append(pred)

        return torch.stack(preds, dim=1)


class DecisionTreeModel:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.name = f"decision_tree_max_depth={max_depth}"

    # inds is a list containing indices where we want the prediction.
    # prediction made at all indices by default.
    def __call__(self, xs, ys, inds=None):
        xs, ys = xs.cpu(), ys.cpu()

        if inds is None:
            inds = range(ys.shape[1])
        else:
            if max(inds) >= ys.shape[1] or min(inds) < 0:
                raise ValueError("inds contain indices where xs and ys are not defined")

        preds = []

        # i: loop over num_points
        # j: loop over bsize
        for i in inds:
            pred = torch.zeros_like(ys[:, 0])

            if i > 0:
                pred = torch.zeros_like(ys[:, 0])
                for j in range(ys.shape[0]):
                    train_xs, train_ys = xs[j, :i], ys[j, :i]

                    clf = tree.DecisionTreeRegressor(max_depth=self.max_depth)
                    clf = clf.fit(train_xs, train_ys)
                    test_x = xs[j, i : i + 1]
                    y_pred = clf.predict(test_x)
                    pred[j] = y_pred[0]

            preds.append(pred)

        return torch.stack(preds, dim=1)


class XGBoostModel:
    def __init__(self):
        self.name = "xgboost"

    # inds is a list containing indices where we want the prediction.
    # prediction made at all indices by default.
    def __call__(self, xs, ys, inds=None):
        xs, ys = xs.cpu(), ys.cpu()

        if inds is None:
            inds = range(ys.shape[1])
        else:
            if max(inds) >= ys.shape[1] or min(inds) < 0:
                raise ValueError("inds contain indices where xs and ys are not defined")

        preds = []

        # i: loop over num_points
        # j: loop over bsize
        for i in tqdm(inds):
            pred = torch.zeros_like(ys[:, 0])
            if i > 0:
                pred = torch.zeros_like(ys[:, 0])
                for j in range(ys.shape[0]):
                    train_xs, train_ys = xs[j, :i], ys[j, :i]

                    clf = xgb.XGBRegressor()

                    clf = clf.fit(train_xs, train_ys)
                    test_x = xs[j, i : i + 1]
                    y_pred = clf.predict(test_x)
                    pred[j] = y_pred[0].item()

            preds.append(pred)

        return torch.stack(preds, dim=1)
