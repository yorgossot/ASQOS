from .energy_level import EnergyLevel
from type_enforced import Enforcer
import sympy


class Transition():
    @Enforcer
    def __init__(self, energy_level_ket : EnergyLevel, energy_level_bra : EnergyLevel, coefficient) -> None:
        if not (hasattr(energy_level_ket, 'associated_component') and hasattr(energy_level_bra, 'associated_component')):
            raise Exception('Both energy levels have to be associated with a component.')
        if energy_level_ket.associated_component != energy_level_bra.associated_component:
            raise Exception('Both energy levels have to be associated with the same component.')
        
        self.energy_level_ket = energy_level_ket
        self.energy_level_bra = energy_level_bra
        self.coefficient = sympy.sympify(coefficient)
        
        self.name = '|'+ energy_level_ket.name + '><' + energy_level_bra.name +'|'

        self.associated_component = energy_level_bra.associated_component
        self.associated_component.add_transition(self)

    def __repr__(self) -> str:
        return f"Transition {self.name} with coefficient {str(self.coefficient)} for component {self.associated_component.name}" 
    
    def delete(self)  -> None :
        self.associated_component.delete_transition(self)



class Decay():
    @Enforcer
    def __init__(self, excited_energy_level : EnergyLevel, decayed_energy_level : EnergyLevel , coefficient ) -> None:
        if not (hasattr(decayed_energy_level, 'associated_component') and hasattr(excited_energy_level, 'associated_component')):
            raise Exception('Both energy levels have to be associated with a component.')
        if decayed_energy_level.associated_component != excited_energy_level.associated_component:
            raise Exception('Both energy levels have to be associated with the same component.')
        if coefficient < 0 :
            raise Exception('Loss rate has to be non-negative') 
        
        self.excited_energy_level = excited_energy_level
        self.decayed_energy_level = decayed_energy_level

        self.energy_level_ket = decayed_energy_level
        self.energy_level_bra = excited_energy_level
        self.coefficient = sympy.sympify(coefficient)

        self.name =  '|'+ self.energy_level_ket.name + '><' + self.energy_level_bra.name +'|' 

        # Update component and excited state
        self.excited_energy_level.add_decayed_energy_level(self.decayed_energy_level)
        self.associated_component = self.energy_level_bra.associated_component
        self.associated_component.add_decay(self)
    
    def __repr__(self) -> str:
        return f"Decay: {self.name} , Coefficient: {str(self.coefficient)} , Associated component: {self.associated_component.name}" 
    
    def delete(self) -> None:
        self.associated_component.delete_decay(self)
        self.excited_energy_level.delete_decayed_energy_level(self.decayed_energy_level)



class Coupling():

    def __init__(self, transition_a : Transition , transition_b : Transition, coefficient ) -> None:
        self.transition_a = transition_a
        self.associated_component_a = transition_a.associated_component
        
        self.transition_b = transition_b
        self.associated_component_b = transition_b.associated_component

        self.coefficient = sympy.sympify(coefficient)

        self.name = transition_a.name + transition_b.name

        self.associated_component_a.add_coupling(self)
        self.associated_component_b.add_coupling(self)
 
    def __repr__(self) -> str:
        return f"Coupling {self.name} with coefficient: {self.coefficient}"

    def delete(self) -> None:
        self.associated_component_a.delete_coupling(self)
        self.associated_component_b.delete_coupling(self)