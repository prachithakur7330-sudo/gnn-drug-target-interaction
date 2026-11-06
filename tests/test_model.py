import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from dti.data import ATOM_FEATURE_DIM, MAX_PROTEIN_LEN
from dti.model import GraphDTA


def _toy_batch(n=5):
    samples = []
    for i in range(n):
        n_atoms = 3 + i
        x = torch.randn(n_atoms, ATOM_FEATURE_DIM)
        ei = torch.stack([torch.arange(n_atoms), torch.roll(torch.arange(n_atoms), 1)])
        samples.append(Data(x=x, edge_index=ei,
                            target_idx=torch.tensor([i % 4]),
                            y=torch.tensor([5.0 + i])))
    return next(iter(DataLoader(samples, batch_size=n)))


def test_forward_shape():
    proteins = torch.randint(0, 20, (4, MAX_PROTEIN_LEN))
    out = GraphDTA()(_toy_batch(5), proteins)
    assert out.shape == (5,)


def test_prediction_depends_on_protein():
    """Same drug, different target should give different predictions."""
    torch.manual_seed(0)
    model = GraphDTA().eval()
    proteins = torch.randint(1, 20, (2, MAX_PROTEIN_LEN))
    x = torch.randn(4, ATOM_FEATURE_DIM)
    ei = torch.stack([torch.arange(4), torch.roll(torch.arange(4), 1)])
    b0 = next(iter(DataLoader([Data(x=x, edge_index=ei, target_idx=torch.tensor([0]))], batch_size=1)))
    b1 = next(iter(DataLoader([Data(x=x, edge_index=ei, target_idx=torch.tensor([1]))], batch_size=1)))
    with torch.no_grad():
        assert not torch.allclose(model(b0, proteins), model(b1, proteins))


def test_one_training_step_reduces_loss_on_a_single_batch():
    torch.manual_seed(0)
    model = GraphDTA()
    proteins = torch.randint(1, 20, (4, MAX_PROTEIN_LEN))
    batch = _toy_batch(8)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.MSELoss()

    first = loss_fn(model(batch, proteins), batch.y.view(-1)).item()
    for _ in range(40):
        opt.zero_grad()
        loss = loss_fn(model(batch, proteins), batch.y.view(-1))
        loss.backward()
        opt.step()
    assert loss.item() < first        # the model can at least overfit one batch
