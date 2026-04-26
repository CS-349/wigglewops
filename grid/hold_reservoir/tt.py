current_pos = engine.current_positions.items():
if len(current_pos)>10: 
    for ticker, pos in current_pos:
        direction = "BUY" if pos.direction == "SELL" else "BUY"
        engine.execute(ticker, pos.pos_volume, direction, timestamp)

        