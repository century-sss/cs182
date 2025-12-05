# analyze_routing.py
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from eval import get_model_from_run
from samplers import get_data_sampler
from tasks import get_task_sampler

def generate_test_data(task_name, data_name, n_dims, degree, n_points, batch_size, seed=42):
    """Generate test data for a specific polynomial degree"""
    torch.manual_seed(seed)
    
    data_sampler = get_data_sampler(data_name, n_dims)
    task_sampler = get_task_sampler(
        task_name, 
        n_dims, 
        batch_size,
        max_degree=degree,
        noise=False
    )
    
    xs = data_sampler.sample_xs(n_points, batch_size)
    task = task_sampler()
    ys = task.evaluate(xs,None)
    
    return xs, ys, degree * torch.ones(batch_size)


def test_fixed_content_varying_position(model, xs, ys, device='cuda'):
    """Test if same content gets different routing at different positions"""
    model.eval()
    B, n_points, d = xs.shape
    
    # Take first example, repeat at all positions
    fixed_x = xs[:1, 0:1, :].expand(B, 1, d)
    fixed_y = ys[:1, 0:1].expand(B, 1)
    
    gates_by_position = []
    
    with torch.no_grad():
        for pos in range(n_points):
            # Insert fixed example at position `pos`
            xs_temp = xs.clone()
            ys_temp = ys.clone()
            xs_temp[:, pos:pos+1, :] = fixed_x
            ys_temp[:, pos:pos+1] = fixed_y
            
            _, gates = model(xs_temp.to(device), ys_temp.to(device), return_gates=True)
            
            # Extract gates at position `pos` for last layer
            pos_gates = gates[-1][:, 2*pos, :]  # x token at position pos
            gates_by_position.append(pos_gates.cpu())
    
    # Stack: [n_points, B, n_experts]
    gates_matrix = torch.stack(gates_by_position, dim=0)
    
    return gates_matrix


def test_position_vs_content(model, xs, ys, device='cuda'):
    """Check if routing depends on position or content via shuffling"""
    model.eval()
    B, n_points, d = xs.shape
    
    with torch.no_grad():
        # Original sequence
        _, gates_original = model(xs.to(device), ys.to(device), return_gates=True)
        
        # Shuffle the order of examples
        perm = torch.randperm(n_points)
        xs_shuffled = xs[:, perm, :]
        ys_shuffled = ys[:, perm]
        
        _, gates_shuffled = model(xs_shuffled.to(device), ys_shuffled.to(device), return_gates=True)
    
    # Compute similarities
    similarities = []
    for layer_idx, (g_orig, g_shuf) in enumerate(zip(gates_original, gates_shuffled)):
        layer_sims = []
        for i in range(n_points):
            j = perm[i].item()
            # Gates for same content at different positions
            orig_gates = g_orig[:, 2*i, :].cpu()  # x token at position i
            shuf_gates = g_shuf[:, 2*j, :].cpu()  # same content at position j
            
            # Cosine similarity
            sim = F.cosine_similarity(orig_gates, shuf_gates, dim=-1).mean().item()
            layer_sims.append(sim)
        
        similarities.append(layer_sims)
    
    return np.array(similarities)  # [n_layers, n_points]


def analyze_gate_entropy(gate_list):
    """Compute entropy of routing decisions"""
    entropies = []
    for gates in gate_list:
        if gates is None:
            continue
        # gates: [B, S, E]
        query_gates = gates[:, ::2, :]  # x tokens only
        eps = 1e-10
        entropy = -(query_gates * torch.log(query_gates + eps)).sum(dim=-1)
        entropies.append(entropy.mean().item())
    
    return entropies


def compute_layer_expert_heatmap(gate_list, average_over_batch=True):
    """
    Compute average routing weights per layer-expert pair
    Returns: [n_layers, n_experts]
    """
    n_layers = len([g for g in gate_list if g is not None])
    n_experts = gate_list[0].shape[-1] if gate_list[0] is not None else 4
    
    heatmap = np.zeros((n_layers, n_experts))
    
    layer_idx = 0
    for gates in gate_list:
        if gates is None:
            continue
        
        query_gates = gates[:, ::2, :]  # [B, n_points, n_experts]
        
        if average_over_batch:
            avg_gates = query_gates.mean(dim=(0, 1)).cpu().numpy()
        else:
            avg_gates = query_gates.mean(dim=1).cpu().numpy()  # [B, n_experts]
        
        heatmap[layer_idx] = avg_gates if average_over_batch else avg_gates.mean(axis=0)
        layer_idx += 1
    
    return heatmap


def visualize_fixed_content_routing(gates_matrix, save_path=None):
    """Visualize routing for same content at different positions"""
    # gates_matrix: [n_points, B, n_experts]
    # Average over batch
    avg_gates = gates_matrix.mean(dim=1).numpy()  # [n_points, n_experts]
    
    plt.figure(figsize=(10, 6))
    sns.heatmap(avg_gates, 
                annot=True, 
                fmt='.3f',
                xticklabels=[f'Expert {i}' for i in range(avg_gates.shape[1])],
                yticklabels=[f'Pos {i}' for i in range(avg_gates.shape[0])],
                cmap='viridis',
                cbar_kws={'label': 'Average Gate Weight'})
    plt.title('Routing for Same Content at Different Positions')
    plt.xlabel('Expert')
    plt.ylabel('Token Position')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def compare_heatmaps_by_degree(model, task_name, data_name, n_dims, n_points, 
                                batch_size, degrees=[1,2,3,4], device='cuda'):
    """Compare layer-expert heatmaps across different polynomial degrees"""
    heatmaps = {}
    
    for degree in degrees:
        xs, ys, _ = generate_test_data(
            task_name, data_name, n_dims, degree, n_points, batch_size
        )
        
        with torch.no_grad():
            _, gates = model(xs.to(device), ys.to(device), return_gates=True)
        
        heatmap = compute_layer_expert_heatmap(gates)
        heatmaps[f'degree_{degree}'] = heatmap
    
    # Plot all heatmaps
    fig, axes = plt.subplots(1, len(degrees), figsize=(5*len(degrees), 6))
    
    # Find global min/max for consistent colorbar
    vmin = min(hm.min() for hm in heatmaps.values())
    vmax = max(hm.max() for hm in heatmaps.values())
    
    for idx, degree in enumerate(degrees):
        ax = axes[idx] if len(degrees) > 1 else axes
        heatmap = heatmaps[f'degree_{degree}']
        
        sns.heatmap(heatmap, 
                    annot=True, 
                    fmt='.3f',
                    xticklabels=[f'E{i}' for i in range(heatmap.shape[1])],
                    yticklabels=[f'L{i}' for i in range(heatmap.shape[0])],
                    cmap='viridis',
                    vmin=vmin,
                    vmax=vmax,
                    ax=ax,
                    cbar=(idx == len(degrees)-1))
        
        ax.set_title(f'Degree {degree}')
        ax.set_xlabel('Expert')
        if idx == 0:
            ax.set_ylabel('Layer')
    
    plt.tight_layout()
    plt.savefig('heatmaps_by_degree.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return heatmaps


def main():
    # Load trained model
    #run_path = "../models/polynomial_regression/moe"
    #model, conf = get_model_from_run(run_path)
    
    #device = 'cuda' if torch.cuda.is_available() else 'cpu'
    #model = model.to(device).eval()
    
    from moe import TransformerModel_var_moe
    import torch
    from tasks import get_task_sampler
    from samplers import get_data_sampler

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = TransformerModel_var_moe(
        num_experts=4,
        n_dims=1,
        n_positions=81,
        n_embd=128,
        n_layer=6,
        n_head=4
    ).to(device)

    ckpt_path = "../models/polynomial_regression/moe/model_0.pt"
    ckpt = torch.load(ckpt_path, map_location=device)

    # strict=False 避免 num_experts/top_k 不一致报错
    missing, unexpected = model.load_state_dict(ckpt)

    print("Missing keys:", missing)
    print("Unexpected keys:", unexpected)
    print(model)
    # Extract config
    task_name = "polynomial_regression"
    data_name = "uniform"
    n_dims = 1
    n_points = 20
    batch_size = 64
    
    print("=" * 80)
    print("ROUTING ANALYSIS")
    print("=" * 80)
    
    # Test 1: Compare heatmaps by degree
    print("\n1. Comparing layer-expert heatmaps across degrees...")
    heatmaps = compare_heatmaps_by_degree(
        model, task_name, data_name, n_dims, n_points, batch_size,
        degrees=[1, 2, 3, 4], device=device
    )
    
    # Check if they're similar
    degree_pairs = [(1,2), (1,3), (1,4), (2,3), (2,4), (3,4)]
    print("\nHeatmap similarities (Frobenius norm of difference):")
    for d1, d2 in degree_pairs:
        diff = np.linalg.norm(heatmaps[f'degree_{d1}'] - heatmaps[f'degree_{d2}'])
        print(f"  Degree {d1} vs {d2}: {diff:.4f}")
    
    # Test 2: Fixed content at varying positions
    print("\n2. Testing same content at different positions...")
    xs, ys, _ = generate_test_data(
        task_name, data_name, n_dims, degree=2, 
        n_points=n_points, batch_size=batch_size
    )
    gates_matrix = test_fixed_content_varying_position(model, xs, ys, device)
    visualize_fixed_content_routing(gates_matrix, 'fixed_content_routing.png')
    
    # Check variance across positions
    position_variance = gates_matrix.var(dim=0).mean(dim=0)  # [n_experts]
    print(f"  Variance across positions per expert: {position_variance}")
    
    # Test 3: Position vs content (shuffling)
    print("\n3. Testing position vs content dependence...")
    similarities = test_position_vs_content(model, xs, ys, device)
    
    print(f"  Average similarity (content follows position): {similarities.mean():.3f}")
    print(f"  If >0.9: routing is position-dependent")
    print(f"  If <0.5: routing is content-dependent")
    
    # Test 4: Entropy analysis
    print("\n4. Gate entropy analysis...")
    _, gates = model(xs.to(device), ys.to(device), return_gates=True)
    entropies = analyze_gate_entropy(gates)
    
    print(f"  Entropy by layer: {entropies}")
    print(f"  Max entropy (uniform): {np.log(4):.3f} for 4 experts")
    print(f"  Low entropy (<0.5): specialized routing")
    print(f"  High entropy (>1.0): uniform routing")
    
    print("\n" + "=" * 80)
    print("Analysis complete! Check generated figures.")
    print("=" * 80)


if __name__ == "__main__":
    main()