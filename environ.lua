local environ = {}

-- States are a tuple of dealer's first card (1-10) and the player's sum
-- Set of actions
environ.A = {'stick', 'hit'}
-- State transitions are not explicitly stored

-- Draws a card (with replacement)
local drawCard = function()
  -- Draw number uniformly from 1-10
  local num = torch.random(1, 10)
  -- Draw colour red with p = 1/3 and black with p = 2/3
  if torch.uniform() <= 1/3 then
    return {num, 'red'}
  else
    return {num, 'black'}
  end
end

-- Adds a card to an existing sum
local addCard = function(sum, card)
  if card[2] == 'black' then
    return sum + card[1]
  else
    return sum - card[1]
  end
end

-- Checks if bust
local checkBust = function(sum)
  return sum > 21 or sum < 1
end

-- Performs a step in the environment
environ.step = function(s, a)
  -- Current state
  local dealersFirstCard = s[1]
  local playersSum = s[2]
  -- Next state (initialised with deep copy of current state)
  local sPrime = {}
  sPrime[1] = dealersFirstCard
  sPrime[2] = playersSum
  -- Reward (0 by default)
  local r = 0

  -- Process actions
  if a == 'hit' then
    -- Draw and add next card
    sPrime[2] = addCard(playersSum, drawCard())

    -- Check player fail conditions
    if checkBust(sPrime[2]) then
      r = -1
    end
  elseif a == 'stick' then
    local dealersSum = dealersFirstCard

    -- Dealer sticks when on 17 or higher
    while dealersSum < 17 and dealersSum >= 1 do
      -- Draw and add next card
      dealersSum = addCard(dealersSum, drawCard())
    end

    -- Check dealer fail conditions and winning conditions otherwise (player with largest sum)
    if checkBust(dealersSum) or dealersSum < playersSum then
      r = 1
    elseif dealersSum > playersSum then
      r = -1
    end
  end

  return sPrime, r
end

-- Calculates if the current state is terminal given previous action and reward
environ.isTerminal = function(a, r)
  return a == 'stick' or r == -1
end

return environ
