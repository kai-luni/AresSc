class KdhStateDto:
    """keep track of  values to calculate the reward"""
    def __init__(self):
        self.destroyed_buildings = None
        self.killed_enemies = None
        self.own_buildings = None
        self.own_army = None
        self.own_money = None

        self.action = None
    
    