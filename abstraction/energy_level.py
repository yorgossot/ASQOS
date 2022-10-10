import sympy
from type_enforced import Enforcer


class EnergyLevel():
    
    def __init__(self, name : str, energy ) -> None:
        assert isinstance(name,str)

        self.name = name
        self.energy = sympy.sympify(energy)
    
        self.stable =  True
        self.decayed_energy_levels = {}

    def __repr__(self) -> str:
        return f"Energy Level {str(self.name)} with energy E={str(self.energy)}, Stable = {self.stable}" 

    def set_energy(self, energy ):
        self.energy = sympy.sympify(energy)
    
    @Enforcer
    def add_decayed_energy_level(self, decayed_energy_level ):
        
        assert isinstance(decayed_energy_level, EnergyLevel)
        
        self.decayed_energy_levels[decayed_energy_level.name] = decayed_energy_level
        self.stable = False
    
    @Enforcer
    def delete_decayed_energy_level(self, decayed_energy_level):
        
        assert isinstance(decayed_energy_level, EnergyLevel)
        
        del self.decayed_energy_levels[decayed_energy_level.name] 
        # if length went to zero, then it became stable again
        if len(self.decayed_energy_levels) == 0:
            self.stable = True
        
    @Enforcer
    def associate_with_component(self, component) -> None:
        from .component import Component
        assert isinstance(component,Component)
        if not hasattr(self, 'associated_component'):
            self.associated_component = component
        else:
            raise Exception(f"Cannot associate {self} with component {component} since it \
                is already associated with {self.associated_component}")
    



