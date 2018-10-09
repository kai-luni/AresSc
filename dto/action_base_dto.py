class ActionBaseDto:
    """actions for building the base"""
    # def __init__():
    #     self.do_nothing_var = 'do_nothing'
    #     self.build_supply_depot_var = 'build_supply_depot'
    #     self.build_barracks_var = 'build_barracks'
    #     self.build_marine_var = 'builld_marine' 

    def get_action_by_index(self, index):
        """for later when a neural net decides"""
        if index == 0:
            return self.do_nothing()
        if index == 1:
            return self.build_supply_depot()
        if index == 2:
            return self.build_barracks()
        if index == 3:     
            return self.build_marine()

    @staticmethod
    def do_nothing():
        return 'do_nothing'

    @staticmethod
    def build_supply_depot():
        return 'build_supply_depot'

    @staticmethod
    def build_command_center():
        return 'builld_command_center'

    @staticmethod
    def build_barracks():
        return 'build_barracks'

    @staticmethod
    def build_marine():
        return 'builld_marine'

    