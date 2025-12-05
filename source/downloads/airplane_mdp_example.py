"""
Airplane MDP Example - Placeholder Code

This is a placeholder Python file demonstrating the structure for the airplane MDP example.
The MDP models a flight ticket selling scenario with 3 customer types and capacity constraints.

Customer Types:
- Type 1: Pays 3000 euros
- Type 2: Pays 2000 euros  
- Type 3: Pays 1000 euros

Flight capacity: Maximum 10 passengers
Selling period: 25 days before flight departure
"""

import numpy as np
from typing import Tuple, Dict

# MDP State Components
# 1. Remaining days until flight
# 2. Number of remaining seats
# 3. Customer type (price: 1000, 2000, or 3000)

def get_state(remaining_days: int, remaining_seats: int, customer_price: int) -> Tuple[int, int, int]:
    """
    Returns the current state of the MDP.
    
    Args:
        remaining_days: Days until flight departure
        remaining_seats: Number of seats still available
        customer_price: Price the current customer is willing to pay
        
    Returns:
        Tuple representing the state (days, seats, price)
    """
    return (remaining_days, remaining_seats, customer_price)


def get_action_space(remaining_seats: int) -> list:
    """
    Returns available actions given the current state.
    
    Args:
        remaining_seats: Number of seats still available
        
    Returns:
        List of available actions (accept or reject)
    """
    if remaining_seats > 0:
        return ['accept', 'reject']
    else:
        return ['reject']  # Cannot accept if no seats available


def calculate_reward(action: str, customer_price: int) -> float:
    """
    Calculates the reward for taking an action.
    Note: DynaPlex uses cost-based rewards (negative values).
    
    Args:
        action: 'accept' or 'reject'
        customer_price: Price the customer is willing to pay
        
    Returns:
        Reward value (negative for costs)
    """
    if action == 'accept':
        return -customer_price  # Negative because DynaPlex is cost-based
    else:
        return 0.0


def simple_policy(remaining_days: int, remaining_seats: int, customer_price: int) -> str:
    """
    Simple rule-based policy for the airplane MDP.
    
    Policy rules:
    1. If more than 5 seats left: sell to all customers
    2. If 1-5 seats and <= 9 days remaining: sell to Type 1 and Type 2 (2000+)
    3. If 1-5 seats and >= 10 days remaining: sell only to Type 1 (3000)
    4. If no seats: reject all
    
    Args:
        remaining_days: Days until flight departure
        remaining_seats: Number of seats still available
        customer_price: Price the customer is willing to pay
        
    Returns:
        Action: 'accept' or 'reject'
    """
    # Rule 4: No seats available
    if remaining_seats == 0:
        return 'reject'
    
    # Rule 1: More than 5 seats left
    if remaining_seats > 5:
        return 'accept'
    
    # Rule 2: 1-5 seats and <= 9 days remaining
    if remaining_seats <= 5 and remaining_days <= 9:
        if customer_price >= 2000:  # Type 1 or Type 2
            return 'accept'
        else:
            return 'reject'
    
    # Rule 3: 1-5 seats and >= 10 days remaining
    if remaining_seats <= 5 and remaining_days >= 10:
        if customer_price == 3000:  # Only Type 1
            return 'accept'
        else:
            return 'reject'
    
    return 'reject'


# Example usage (placeholder)
if __name__ == "__main__":
    # Example state: 15 days remaining, 3 seats left, Type 2 customer (2000)
    state = get_state(remaining_days=15, remaining_seats=3, customer_price=2000)
    print(f"Current state: {state}")
    
    # Get available actions
    actions = get_action_space(remaining_seats=3)
    print(f"Available actions: {actions}")
    
    # Apply policy
    action = simple_policy(remaining_days=15, remaining_seats=3, customer_price=2000)
    print(f"Policy decision: {action}")
    
    # Calculate reward
    reward = calculate_reward(action, customer_price=2000)
    print(f"Reward: {reward}")

