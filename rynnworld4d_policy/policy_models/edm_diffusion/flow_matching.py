"""
Flow Matching policy wrapper for DiffusionTransformer.

Modified from: https://github.com/roboterax/video-prediction-policy
Flow matching formulation based on Lipman et al. "Flow Matching for Generative Modeling" (2023).

Replaces EDM (Karras) score matching with conditional flow matching:
  - Training: linear interpolation x_t = (1-t)*noise + t*data, predict velocity v = data - noise
  - Inference: Euler ODE solver with N steps (default 4)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import einops

from policy_models.module.diffusion_decoder import DiffusionTransformer


class FlowMatchingPolicy(nn.Module):
    """Flow matching wrapper around DiffusionTransformer.

    Drop-in replacement for GCDenoiser: same inner_model architecture,
    but uses flow matching loss and Euler ODE sampling instead of EDM.
    """

    def __init__(
        self,
        action_dim,
        obs_dim,
        goal_dim,
        num_tokens,
        goal_window_size,
        obs_seq_len,
        act_seq_len,
        device,
        proprio_dim=8,
    ):
        super().__init__()
        self.inner_model = DiffusionTransformer(
            action_dim=action_dim,
            obs_dim=obs_dim,
            goal_dim=goal_dim,
            proprio_dim=proprio_dim,
            goal_conditioned=True,
            embed_dim=384,
            n_dec_layers=4,
            n_enc_layers=4,
            n_obs_token=num_tokens,
            goal_seq_len=goal_window_size,
            obs_seq_len=obs_seq_len,
            action_seq_len=act_seq_len,
            embed_pdrob=0,
            goal_drop=0,
            attn_pdrop=0.3,
            resid_pdrop=0.1,
            mlp_pdrop=0.05,
            n_heads=8,
            device=device,
            use_mlp_goal=True,
        )
        self.action_dim = action_dim
        self.act_seq_len = act_seq_len

    def _encode_time(self, t):
        """Encode t in [0,1] → embedding, reusing sigma_emb's sinusoidal + MLP."""
        # Scale t to a range where sinusoidal features are informative.
        # EDM uses log(sigma)/4 which spans roughly [-2, 5].
        # We map t in [0,1] to [0, 5] to get similar coverage.
        t_scaled = t * 5.0
        t_scaled = einops.rearrange(t_scaled, "b -> b 1")
        emb = self.inner_model.sigma_emb(t_scaled)
        if len(emb.shape) == 2:
            emb = einops.rearrange(emb, "b d -> b 1 d")
        return emb

    def predict_velocity(self, state, x_t, goal, t):
        """Forward: predict velocity field v(x_t, t).

        Args:
            state: dict with 'state_images' (B, N, D) and optionally 'state_obs' (B, proprio_dim)
            x_t: noisy actions (B, T, action_dim)
            goal: language goal (B, 1, goal_dim)
            t: flow time (B,) in [0, 1]
        Returns:
            predicted velocity (B, T, action_dim)
        """
        emb_t = self._encode_time(t)

        goals = self.inner_model.preprocess_goals(goal, state["state_images"].size(1))
        state_embed, proprio_embed = self.inner_model.process_state_embeddings(state)
        if proprio_embed is not None and proprio_embed.dim() == 2:
            proprio_embed = proprio_embed.unsqueeze(1)  # (B, D) -> (B, 1, D)
        goal_embed = self.inner_model.process_goal_embeddings(goals)

        input_seq = self.inner_model.concatenate_inputs(
            emb_t, goal_embed, state_embed, proprio_embed
        )
        context = self.inner_model.encoder(input_seq)

        action_embed = self.inner_model.action_emb(x_t)
        action_x = self.inner_model.drop(action_embed)
        x = self.inner_model.decoder(action_x, emb_t, context)
        return self.inner_model.action_pred(x)

    def loss(self, state, actions, goal):
        """Compute flow matching loss.

        Samples t ~ U(0,1), constructs x_t via linear interpolation,
        trains the model to predict the velocity field v = data - noise.
        """
        B = actions.shape[0]
        t = torch.rand(B, device=actions.device).clamp(1e-4, 1.0)
        noise = torch.randn_like(actions)

        t_expand = t.view(B, 1, 1)
        x_t = (1 - t_expand) * noise + t_expand * actions
        v_target = actions - noise

        v_pred = self.predict_velocity(state, x_t, goal, t)
        return F.mse_loss(v_pred, v_target), v_pred

    def forward(self, state, action, goal, sigma, **kwargs):
        """Compatibility forward — interprets sigma as t for flow matching."""
        return self.predict_velocity(state, action, goal, sigma)

    @torch.no_grad()
    def sample(self, state, goal, shape, n_steps=4):
        """Generate actions via Euler ODE integration.

        Integrates from t=0 (noise) to t=1 (data) in n_steps.
        """
        device = goal.device
        B = shape[0]
        x = torch.randn(shape, device=device)
        dt = 1.0 / n_steps

        for i in range(n_steps):
            t = torch.full((B,), i * dt, device=device)
            v = self.predict_velocity(state, x, goal, t)
            x = x + v * dt

        return x

    def get_params(self):
        return self.inner_model.parameters()
