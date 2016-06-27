local _ = require 'moses'
local nn = require 'nn'
local gnuplot = require 'gnuplot'
local environ = require 'environ'

-- Set manual seed
torch.manualSeed(1)

-- Load Q* from MC control
local QStar = torch.load('Q.t7')
-- Extract V as argmax Q
local V = torch.max(QStar, 3):squeeze()

local nEpisodes = 1000000
-- Number of discrete actions
local m = #environ.A

-- Initial exploration ɛ
local epsilon = 1
-- Linear ɛ decay
local epsilonDecay = 1/nEpisodes
-- Minimum ɛ
local epsilonMin = 0.05
-- Constant step-size ɑ
local alpha = 0.001
-- RMSProp decay
local decay = 0.9
-- (Machine) epsilon
local eps = 1e-20
-- Entropy regularisation factor β
local beta = 0.01

-- Create policy network π
local net = nn.Sequential()
net:add(nn.Linear(2, 16))
net:add(nn.ReLU(true))
net:add(nn.Linear(16, m))
net:add(nn.SoftMax())
-- Get network parameters θ
local theta, gradTheta = net:getParameters()
-- Moving average of squared gradient
local gradThetaSq = torch.Tensor(gradTheta:size()):zero()

-- Results from each episode
local results = torch.Tensor(nEpisodes)

-- Sample
for i = 1, nEpisodes do
  -- Experience tuples (s, a, r)
  local E = {}
  -- Pick random starting state
  local s = {torch.random(1, 10), torch.random(1, 21)}
  
  -- Run till termination
  repeat
    -- Choose action by ɛ-greedy exploration
    local aIndex
    if torch.uniform() < (1 - epsilon) then -- Exploit with probability 1 - ɛ
      -- Get categorical action distribution from π = p(s; θ)
      local probs = net:forward(torch.Tensor(s))
      probs:add(eps) -- Add small probability to prevent NaNs
      -- Sample action ~ p(s; θ)
      aIndex = torch.multinomial(probs, 1)[1]
    else
      -- Otherwise pick any action with probability 1/m
      aIndex = torch.random(1, m)
    end
    local a = environ.A[aIndex]

    -- Perform a step
    local sPrime, r = environ.step(s, a) -- r comes from score function f(s)

    -- Store experience tuple
    table.insert(E, {s, a, r})

    -- Set next state as current state
    s = sPrime

    -- Linearly decay ɛ
    epsilon = math.max(epsilon - epsilonDecay, epsilonMin)
  until environ.isTerminal(a, r)
  
  -- Save result of episode
  results[i] = E[#E][3]
  
  -- Reset ∇θ
  gradTheta:zero()
  
  -- Learn from experience of one complete episode
  for j = 1, #E do
    -- Extract experience
    local s = E[j][1]
    local a = E[j][2]
    -- Get action index
    local aIndex = _.find(environ.A, a)

    -- Calculate variance-reduced reward (advantage) ∑t r - b(s) = ∑t r - V(s) = A
    local A = 0
    for k = j, #E do
      A = A + (E[k][3] - V[s[1]][s[2]])
    end

    -- Use a policy gradient update (REINFORCE rule): ∇θ Es[f(s)] = ∇θ ∑s p(s)f(s) = Es[f(s) ∇θ logp(s)]
    local input = torch.Tensor(s)
    local output = net:forward(input)
    output:add(eps) -- Add small probability to prevent NaNs

    -- ∇θ logp(s) = 1/p(a) for chosen a, 0 otherwise
    local target = torch.zeros(m)
    target[aIndex] = A * 1/output[aIndex] -- f(s) ∇θ logp(s)

    -- Calculate gradient of entropy of policy: -∑a logp(s) + 1
    local gradEntropy = -torch.sum(torch.log(output) + 1)
    -- Add to target to improve exploration (prevent convergence to suboptimal deterministic policy)
    target:add(beta * gradEntropy)
    
    -- Accumulate gradients
    net:backward(input, target)
  end

  -- Update moving average of squared gradients
  gradThetaSq = decay * gradThetaSq + (1 - decay) * torch.pow(gradTheta, 2)
  -- RMSProp update (gradient ascent version)
  theta:add(torch.cdiv(alpha * gradTheta, torch.sqrt(gradThetaSq) + eps))
end

-- Take average results over 1000 episodes
local avgResults = torch.Tensor(nEpisodes/1000)
for ep = 1, nEpisodes, 1000 do
  avgResults[(ep - 1)/1000 + 1] = torch.mean(results:narrow(1, ep, 1000))
end

-- Plot results
gnuplot.pngfigure('PolicyGradient.png')
gnuplot.plot('Average Result', torch.linspace(1, nEpisodes/1000, nEpisodes/1000), avgResults)
gnuplot.title('Policy Gradient Results')
gnuplot.ylabel('Result (Mean over 1000 Episodes)')
gnuplot.xlabel('Episode (x1000)')
gnuplot.plotflush()
