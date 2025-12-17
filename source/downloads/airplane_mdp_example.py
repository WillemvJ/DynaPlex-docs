"""
This demonstrates the definition of an MDP and a policy in DynaML - the dynamic modelling language
that supports DynaPlex 2.0. 
Defining an MDP like this is a starting point for various algorithms. 
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
    price_offered_per_seat: int
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
    prices_per_customer_type: list[int]
    customer_type_probs: list[float]
    
    def __init__(
        self,
        initial_days: int,
        initial_seats: int,
        prices_per_customer_type: list[int],
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
        # NOTE:  __init__ on the MDP class itself is never called by the dynaplex compiler, so we can use any valid cpython code here
        #  but the class must be a dataclass, and other functions need to be valid  DynaML code.  
        
        # Validating parameters
        assert initial_days > 0 and initial_seats > 0
        assert prices_per_customer_type and all(price > 0 for price in prices_per_customer_type)
        assert customer_type_probs and len(customer_type_probs) == len(prices_per_customer_type)
        assert all(prob >= 0 for prob in customer_type_probs) and np.isclose(sum(customer_type_probs), 1.0, atol=1e-6)
        
        # Set attributes
        #NOTE: only set attributes that are part of the annotation.
        self.initial_days = initial_days
        self.initial_seats = initial_seats
        self.prices_per_customer_type = prices_per_customer_type
        self.customer_type_probs = customer_type_probs
    
    def get_initial_state(self, rng: Generator) -> State:
        """
        Generates and returns an initial state of the MDP.
        
        Args:
            rng: NumPy random generator to support random initial state.
        """
        # NOTE: function get_initial_state and any functions that it calls must be valid DynaML code.
        return State(
            remaining_days=self.initial_days,
            remaining_seats=self.initial_seats,
            price_offered_per_seat=0,
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
        # NOTE: function modify_state_with_event and any functions that it calls must be valid DynaML code.

        # Sample customer type from discrete distribution
        # So rng.choice works the same as np.random.choice; it essentially is the 
        # modern/recommended way to generate random numbers in numpy. 
        # NOTE: Never use np.random directly, use the rng parameter instead!
        customer_type = rng.choice(
            len(self.customer_type_probs),
            p=self.customer_type_probs,
        )
        state.price_offered_per_seat = self.prices_per_customer_type[customer_type]        
        
        # After processing event, we await an action - 
        # the agent must decide whether to accept or reject the customer.
        state.category = StateCategory.AWAIT_ACTION        
        return 0.0  # No cost for event transitions
    
    def modify_state_with_action(self, state: State, action: int) -> float:
        """
        Apply an action to the state (modify in place).
        
        Args:
            state: Current state (modified in place)
            action: Action to take (0=reject, 1=accept)
            
        Returns:
            Cost incurred (negative revenue for accepted customers)
        """
        # NOTE: this function (and any functions that it calls) must be valid DynaML code.

        # NOTE: do _not_ attempt to generate random numbers here. Any random transitions must happen 
        # in modify_state_with_event, using the rng parameter passed in there. 
        
        assert state.remaining_days > 0 
        state.remaining_days -= 1

        # After processing action, we await the next event - customer arrival. 
        if state.remaining_days == 0:
            state.category = StateCategory.FINAL
        else:
            state.category = StateCategory.AWAIT_EVENT

        if action == 0:
            # Reject customer
            state.price_offered_per_seat = 0
            return 0.0
        
        elif action == 1:
            # Accept customer
            # Sell the seat to the customer. 
            state.remaining_seats -= 1            
            # Use a cost-based formulation (cost = -reward)
            cost = -state.price_offered_per_seat
            # Reset the price offered per seat to 0, awaiting the next event. 
            state.price_offered_per_seat = 0
            return cost
        
        else:
            assert False, f"Invalid action: Must be 0 (reject) or 1 (accept)"
    
    def is_allowed_action(self, state: State, action: int) -> bool:
        """
        Check if an action is allowed in the current state.

        Returns:
            True if action is allowed
        """
        # NOTE: function is_allowed_action must be valid DynaML code.   

        if action == 0:
            return True
        elif action == 1:
            return state.remaining_seats > 0
        else:
            assert False, f"Invalid action: Must be 0 (reject) or 1 (accept)"


@dataclass
class SimplePolicy:
    """
    Simple rule-based policy for the airplane MDP. This policy adheres to the DynaPlex DSL. 
    
    This policy uses threshold-based rules to decide when to accept or reject customers.   
 
    """
    mdp: AirplaneMDP
    seat_threshold: int = 5
    days_threshold: int = 9
    min_price_low_days: int = 2000
    min_price_high_days: int = 3000
    
    def get_action(self, state: State) -> int:
        """
        Determine which action to take given the current state. Simple heuristic policy.


        # NOTE: this function must be valid DynaML code.
        Args:
            state: Current state
            
        Returns:
            Action (0=reject, 1=accept)
        """
        if state.remaining_seats == 0:
            return 0
        
        # Rule 1: More than seat_threshold seats left
        if state.remaining_seats > self.seat_threshold:
            return 1
        
        # Rule 2: 1-seat_threshold seats and <= days_threshold remaining
        if state.remaining_days <= self.days_threshold and state.price_offered_per_seat >= self.min_price_low_days:
            return 1
        
        # Rule 3: 1-seat_threshold seats and > days_threshold remaining
        if state.remaining_days > self.days_threshold and state.price_offered_per_seat >= self.min_price_high_days:
            return 1
        
        return 0


def simulate_episode(mdp: AirplaneMDP, policy: SimplePolicy, *, seed: int = 42, verbose: bool = True) -> float:
    """
    Simulate a single episode using the given policy.

    This loop illustrates the interaction between the MDP, state, and policy. It is not needed for the API.
    
    The simulation loop continues until state.category == FINAL.
    On each iteration, it dispatches based on the state category:
    - AWAIT_EVENT: call modify_state_with_event
    - AWAIT_ACTION: call modify_state_with_action
    
    Args:
        mdp: MDP instance
        policy: Policy instance to use for action selection
        seed: Random seed (keyword-only)
        verbose: Whether to print detailed output during simulation (keyword-only)
        
    Returns:
        Total cost (negative revenue) for the episode
    """
    # NOTE: this function is just a cpython function used to simulate the MDP.
    # It is not needed for the API. 

    rng = np.random.default_rng(seed)
    state = mdp.get_initial_state(rng)
    
    total_cost = 0.0
    step = 0
    
    if verbose:
        print(f"Initial state: {state}")
        print("-" * 80)
    
    while state.category != StateCategory.FINAL:
        if state.category == StateCategory.AWAIT_EVENT:
            # Generate customer arrival event
            cost = mdp.modify_state_with_event(state, rng)
            total_cost += cost
            if verbose:
                print(f"  State after event: {state}")
            
        elif state.category == StateCategory.AWAIT_ACTION:
            # Apply policy and execute action
            action = policy.get_action(state)
            
            if verbose:
                action_name = "ACCEPT" if action == 1 else "REJECT"
                print(f"Step {step}: ACTION {action_name} ({action})")
            
            cost = mdp.modify_state_with_action(state, action)
            total_cost += cost
            
            if verbose:
                print(f"  State after action: {state}")
            step += 1
        
        else:
            raise RuntimeError(f"Unexpected state category: {state.category}")
    
    if verbose:
        print("-" * 80)
        print(f"Episode finished: {step} steps, total revenue: €{-total_cost:.0f}")
    
    return total_cost


def main() -> None:
    """Run airplane MDP simulation example."""
    # Create MDP with standard configuration
    mdp = AirplaneMDP(
        initial_days=25,
        initial_seats=10,
        prices_per_customer_type=[3000, 2000, 1000],
        customer_type_probs=[0.4, 0.3, 0.3],
    )
    
    # Create policy with default parameters
    policy = SimplePolicy(mdp=mdp)

    # Run single simulation with detailed output
    print("=" * 80)
    print("DETAILED SIMULATION (Single Episode)")
    print("=" * 80)
    simulate_episode(mdp, policy, seed=42, verbose=True)
    
    # Run 1000 simulations to estimate average performance
    print("\n" + "=" * 80)
    print("PERFORMANCE EVALUATION (1000 Episodes)")
    print("=" * 80)
    
    num_simulations = 1000
    total_costs = []
    
    for i in range(num_simulations):
        cost = simulate_episode(mdp, policy, seed=i, verbose=False)
        total_costs.append(cost)
    
    # Calculate statistics (remember: cost = -revenue, so profit = -cost)
    average_cost = np.mean(total_costs)
    average_profit = -average_cost
    # standard error of the mean
    std_error = np.std(total_costs) / np.sqrt(num_simulations)
    min_profit = -np.max(total_costs)
    max_profit = -np.min(total_costs)
    
    print(f"Number of simulations: {num_simulations}")
    print(f"Average profit: €{average_profit:.2f}")
    print(f"Standard error of the mean: €{std_error:.2f}")
    print(f"Min profit: €{min_profit:.2f}")
    print(f"Max profit: €{max_profit:.2f}")
    print("=" * 80)

if __name__ == "__main__":
    main()