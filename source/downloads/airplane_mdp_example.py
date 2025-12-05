"""
Airplane MDP Example

This demonstrates the Python API for defining an MDP in DynaPlex 2.
The airplane MDP models a flight ticket selling scenario with capacity constraints.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto

import numpy as np
from numpy.random import Generator


class StateCategory(Enum):
    """State category determines which operation should be called next."""
    AWAIT_EVENT = auto()
    AWAIT_ACTION = auto()
    FINAL = auto()


@dataclass
class State:
    """
    State representation for the airplane MDP.
    """
    remaining_days: int
    remaining_seats: int
    price_offered_per_seat: float
    # this member must always be defined on any dynaplex MDP state:
    category: StateCategory = StateCategory.AWAIT_EVENT
    

@dataclass(init=False)
class AirplaneMDP:
    """
    Airplane ticket selling MDP.
    
    Actions:
        0: Reject customer
        1: Accept customer (sell seat)
    """
    
    # MDP configuration (instance attributes, no defaults)
    initial_days: int
    initial_seats: int
    prices_per_customer_type: list[float]
    customer_type_probs: list[float]
    
    def __init__(
        self,
        initial_days: int,
        initial_seats: int,
        prices_per_customer_type: list[float],
        customer_type_probs: list[float],
    ):
        """
        Initialize the Airplane MDP with validation.
        
        Args:
            initial_days: Number of days in selling period (must be > 0)
            initial_seats: Flight capacity (must be > 0)
            prices_per_customer_type: List of prices for each customer type (all must be > 0)
            customer_type_probs: Probability distribution over customer types (must sum to 1.0)
            
        Raises:
            ValueError: If any validation checks fail
        """
        # NOTE:  __init__ is never called by the dynaplex compiler, so we can use any valid cpython code here. 
        


        # Validate initial_days
        if initial_days <= 0:
            raise ValueError(f"initial_days must be positive, got {initial_days}")
        
        # Validate initial_seats
        if initial_seats <= 0:
            raise ValueError(f"initial_seats must be positive, got {initial_seats}")
        
        # Validate prices_per_customer_type
        if not prices_per_customer_type:
            raise ValueError("prices_per_customer_type cannot be empty")
        if any(price <= 0 for price in prices_per_customer_type):
            raise ValueError(f"All prices must be positive, got {prices_per_customer_type}")
        
        # Validate customer_type_probs
        if not customer_type_probs:
            raise ValueError("customer_type_probs cannot be empty")
        if len(customer_type_probs) != len(prices_per_customer_type):
            raise ValueError(
                f"Length mismatch: {len(customer_type_probs)} probabilities "
                f"for {len(prices_per_customer_type)} customer types"
            )
        if any(prob < 0 for prob in customer_type_probs):
            raise ValueError(f"All probabilities must be non-negative, got {customer_type_probs}")
        
        prob_sum = sum(customer_type_probs)
        if not np.isclose(prob_sum, 1.0, atol=1e-6):
            raise ValueError(
                f"Probabilities must sum to 1.0, got {prob_sum} "
                f"(probabilities: {customer_type_probs})"
            )
        
        # All validations passed, set attributes
        #NOTE: only set attributes that are part of the annotation.
        self.initial_days = initial_days
        self.initial_seats = initial_seats
        self.prices_per_customer_type = prices_per_customer_type
        self.customer_type_probs = customer_type_probs
    
    def get_initial_state(self, rng: Generator) -> State:
        """
        Generate the initial state of the MDP.
        
        Args:
            rng: NumPy random generator to support random initial state.
            
        Returns:
            Initial state for the MDP. 
        """
        # NOTE: function get_initial_state and any functions that it calls must use only
        # constructs that are part of the DynaPlex DSL.       
        # all used and constructed classes must be dataclasses. etc. 

        return State(
            remaining_days=self.initial_days,
            remaining_seats=self.initial_seats,
            price_offered_per_seat=0.0,
            category=StateCategory.AWAIT_EVENT,
        )
    
    def modify_state_with_event(self, state: State, rng: Generator) -> float:
        """
        Generate a (customer arrival) event and modify state in place.
       
        Args:
            state: Current state (modified in place)
            rng: NumPy random generator for sampling customer type
            
        Returns:
            Cost incurred (in this model, no cost for event transitions)
        """
        # NOTE: function modify_state_with_event and any functions that it calls must use only
        # constructs that are part of the DynaPlex DSL.  

        # Sample customer type from discrete distribution
        # NOTE: Never use np.random directly, use the rng parameter instead!
        customer_type = rng.choice(
            len(self.customer_type_probs),
            p=self.customer_type_probs,
        )
        state.price_offered_per_seat = self.prices_per_customer_type[customer_type]
        
        
        # Check termination conditions
        if state.remaining_days == 0 or state.remaining_seats == 0:
            state.category = StateCategory.FINAL
        else:
            state.category = StateCategory.AWAIT_ACTION

        
        return 0.0  # No cost for event transitions
    
    def modify_state_with_action(self, state: State, action: int) -> float:
        """
        Apply an action to the state (modify in place).
        
        After this method, state.category will be AWAIT_EVENT.
        
        Args:
            state: Current state (modified in place)
            action: Action to take (0=reject, 1=accept)
            
        Returns:
            Cost incurred (negative revenue for accepted customers)
            
        Raises:
            ValueError: If action is invalid or sells when no seats available
        """
        # NOTE: function modify_state_with_action and any functions that it calls must use only
        # constructs that are part of the DynaPlex DSL.  

        # NOTE: do not attempt to generate random numbers here. And random transitiions must happen 
        # in modify_state_with_event, using the rng parameter passed in there. 

        if state.remaining_days == 0:
            raise ValueError("Cannot take action when no days remain")
        
        # After processing action, we await an event
        state.category = StateCategory.AWAIT_EVENT
        
        if action == 0:
            # Reject customer
            state.remaining_days -= 1
            state.price_offered_per_seat = 0.0
            return 0.0
        
        elif action == 1:
            # Accept customer
            if state.remaining_seats <= 0:
                raise ValueError("Cannot sell seat when none available")
            
            state.remaining_seats -= 1
            state.remaining_days -= 1
            
            # DynaPlex uses cost-based formulation (cost = -reward)
            cost = -state.price_offered_per_seat
            state.price_offered_per_seat = 0.0
            
            return cost
        
        else:
            raise ValueError(
                f"Invalid action: {action}. Must be 0 (reject) or 1 (accept)"
            )
    
    def is_allowed_action(self, state: State, action: int) -> bool:
        """
        Check if an action is allowed in the current state.
        
        Args:
            state: Current state
            action: Action to check
            
        Returns:
            True if action is allowed
        """
        # NOTE: function is_allowed_action and any functions that it calls must use only
        # constructs that are part of the DynaPlex DSL.  

        if action == 0:
            return True
        elif action == 1:
            return state.remaining_seats > 0
        else:
            return False


@dataclass
class SimplePolicy:
    """
    Simple rule-based policy for the airplane MDP. This policy adheres to the DynaPlex DSL. 
    
    This policy uses threshold-based rules to decide when to accept or reject customers.   
 
    """
    mdp: AirplaneMDP
    seat_threshold: int = 5
    days_threshold: int = 9
    min_price_low_days: float = 2000.0
    min_price_high_days: float = 3000.0
    
    def get_action(self, state: State) -> int:
        """
        Determine which action to take given the current state.
        
        Policy rules:
        1. If more than seat_threshold seats left: sell to all customers
        2. If 1-seat_threshold seats and <= days_threshold remaining: 
           sell to customers paying >= min_price_low_days
        3. If 1-seat_threshold seats and > days_threshold remaining: 
           sell to customers paying >= min_price_high_days
        4. If no seats: reject all
        
        Args:
            state: Current state
            
        Returns:
            Action (0=reject, 1=accept)
        """
        # Rule 4: No seats available
        if state.remaining_seats == 0:
            return 0
        
        # Rule 1: More than seat_threshold seats left
        if state.remaining_seats > self.seat_threshold:
            return 1
        
        # Rule 2: 1-seat_threshold seats and <= days_threshold remaining
        if state.remaining_days <= self.days_threshold:
            return 1 if state.price_offered_per_seat >= self.min_price_low_days else 0
        
        # Rule 3: 1-seat_threshold seats and > days_threshold remaining
        if state.remaining_days > self.days_threshold:
            return 1 if state.price_offered_per_seat >= self.min_price_high_days else 0
        
        return 0


def simulate_episode(mdp: AirplaneMDP, policy: SimplePolicy, *, seed: int = 42) -> float:
    """
    Simulate a single episode using the given policy.
    
    The simulation loop continues until state.category == FINAL.
    On each iteration, it dispatches based on the state category:
    - AWAIT_EVENT: call modify_state_with_event
    - AWAIT_ACTION: call modify_state_with_action
    
    Args:
        mdp: MDP instance
        policy: Policy instance to use for action selection
        seed: Random seed (keyword-only)
        
    Returns:
        Total cost (negative revenue) for the episode
    """
    # NOTE: this function is just a cpython function used to simulate the MDP.
    # It is not needed for the API. 

    rng = np.random.default_rng(seed)
    state = mdp.get_initial_state(rng)
    
    total_cost = 0.0
    step = 0
    
    print(f"Initial state: {state}")
    print("-" * 80)
    
    while state.category != StateCategory.FINAL:
        if state.category == StateCategory.AWAIT_EVENT:
            # Generate customer arrival event
            cost = mdp.modify_state_with_event(state, rng)
            total_cost += cost
            print(f"  State after event: {state}")
            
        elif state.category == StateCategory.AWAIT_ACTION:
            # Apply policy and execute action
            action = policy.get_action(state)
            action_name = "ACCEPT" if action == 1 else "REJECT"
            
            print(f"Step {step}: ACTION {action_name} ({action})")
            
            cost = mdp.modify_state_with_action(state, action)
            total_cost += cost
            print(f"  State after action: {state}")
            step += 1
        
        else:
            raise RuntimeError(f"Unexpected state category: {state.category}")
    
    print("-" * 80)
    print(f"Episode finished: {step} steps, total revenue: €{-total_cost:.0f}")
    
    return total_cost


def main() -> None:
    """Run airplane MDP simulation example."""
    # Create MDP with standard configuration
    mdp = AirplaneMDP(
        initial_days=25,
        initial_seats=10,
        prices_per_customer_type=[3000.0, 2000.0, 1000.0],
        customer_type_probs=[0.4, 0.3, 0.3],
    )
    
    # Create policy with default parameters
    policy = SimplePolicy(mdp=mdp)

    #eventually we will be able to run DCL and other algorithms (in C++ backend) based on this:
    #agent_list = dynaplex.train_dcl(mdp, policy)
    
 
    # Run simulation
    simulate_episode(mdp, policy, seed=42)
    
    print()
    print("=" * 60)
    
    # Example: Custom policy configuration
    print("\n\nExample with custom policy:")
    print("=" * 60)
    aggressive_policy = SimplePolicy(
        mdp=mdp,
        seat_threshold=3,
        days_threshold=5,
        min_price_low_days=1500.0,
        min_price_high_days=2500.0,
    )
    print("Aggressive policy (accepts more customers):")
    print(f"  Seat threshold: {aggressive_policy.seat_threshold}")
    print(f"  Min price (low days): €{aggressive_policy.min_price_low_days:.0f}")
    simulate_episode(mdp, aggressive_policy, seed=42)
    print("=" * 60)
    
   
    
 

if __name__ == "__main__":
    main()