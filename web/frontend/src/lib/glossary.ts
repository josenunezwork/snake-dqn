// Just-in-time explanations for the RL jargon scattered across the dense UI, so
// the dashboard teaches itself. Keyed strings referenced by <InfoDot term=…>.
export const GLOSSARY: Record<string, string> = {
  qvalue:
    "Q-value: the network's predicted long-term reward for each action from the current state. The highest Q is the move it takes.",
  margin:
    "Decision margin: how far the best action's Q leads the runner-up. Large = confident; near-zero = a coin-flip.",
  epsilon:
    "ε (epsilon): exploration rate — with probability ε the agent picks a random action instead of the greedy one. 0 = pure exploitation.",
  loss:
    "Training loss: the temporal-difference error the learner minimises. It only moves in Train mode; lower generally means a better-fit value function.",
  freespace:
    "Free-space: how much open room lies in each turn direction (L/S/R). Low values warn that the move would trap the snake in itself.",
  perception:
    "The agent's egocentric view — food density and obstacle danger across 16 directional sectors around its head.",
  dueling:
    "Dueling DQN: the head splits Q into a state-value V(s) plus per-action advantages A(s,a):  Q = V + (A − mean A).",
  bestlen: "Best length: the longest any snake has grown so far this episode.",
  kills: "Kills: head-to-body collisions this snake caused (the victim dies, the attacker is rewarded).",
  alive: "Alive: how many snakes are currently alive in the arena this episode.",
  food: "Food eaten: pellets consumed this episode, summed across all snakes.",
  actiontaken:
    "Action taken: the move the agent actually executed last step — after masking illegal moves and ε-exploration. It can differ from the greedy (highest-Q) pick.",
  massintegral:
    "Mass integral: the promotion gate's headline metric — mean per-frame snake mass over the TOTAL eval horizon, with dead frames counting 0. Rewards growing and staying alive; dying rich no longer outranks surviving.",
};
