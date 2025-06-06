def format_float(value, decimals=4):
    return round(value, decimals)

def payoff_diagram(S_range, K, option_type='call', position='long'):
    payoff = []
    for S in S_range:
        if option_type == 'call':
            val = max(S - K, 0)
        else:
            val = max(K - S, 0)
        payoff.append(val if position == 'long' else -val)
    return payoff
