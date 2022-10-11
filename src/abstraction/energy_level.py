import sympy
from type_enforced import Enforcer




class EnergyLevel():
    
    def __init__(self, name : str, energy ) -> None:
        assert isinstance(name,str)

        self.name = name
        self.energy = sympy.sympify(energy)
    
        self.excitation_status =  None
        self.index_in_component = None
        
        self.rabi_energy_levels = {}
        self.decayed_energy_levels = {}
        self.decaying_energy_levels = {}
        self.coupled_energy_levels = {}

    def __repr__(self) -> str:
        return f"Energy Level {str(self.name)} with energy E={str(self.energy)}" 
    
    
    def _update_excitation_status(self) -> None:
        """
        Updates the excitation status of the energy level
        according to the number of decayed, coupled and rabi levels.
        The excitation status attribute will be set as:
        - `'E'` if the state is excited and can decay
        - `'R'` elif it is involved in a Rabi transition
        - `'C'` elif it is involved in a Coupling
        - `'D'` elif it is a decayed state
        - `'N'` else (non-participating in any interaction)
        
        Parameters
        ----------
        self : EnergyLevel

        Returns
        -------
        None
        """
        if len(self.decayed_energy_levels) > 0:
            self.excitation_status = 'E'
        elif len(self.rabi_energy_levels) > 0:
            self.excitation_status = 'R'
        elif len(self.coupled_energy_levels) > 0:
            self.excitation_status = 'C'
        elif len(self.decaying_energy_levels):
            self.excitation_status = 'D'
        else:
            self.excitation_status = 'N'
            


    def _associate_with_component(self, component) -> None:
        from .component import Component
        assert isinstance(component,Component)
        if not hasattr(self, 'associated_component'):
            self.associated_component = component
        else:
            raise Exception(f"Cannot associate {self} with component {component} since it \
                is already associated with {self.associated_component}")


    def _add_rabi_energy_level(self, rabi_energy_level )-> None:
        
        assert isinstance(rabi_energy_level, EnergyLevel)
        assert rabi_energy_level != self
        assert rabi_energy_level.associated_component == self.associated_component
        
        self.rabi_energy_levels[rabi_energy_level.name] = rabi_energy_level

    

    def _add_decaying_energy_level(self, decaying_energy_level )-> None:
        '''
        Adds an energy level that decays into instance of energy level.
        '''
        assert isinstance(decaying_energy_level, EnergyLevel)
        assert decaying_energy_level != self
        assert decaying_energy_level.associated_component == self.associated_component

        self.decaying_energy_levels[decaying_energy_level.name] = decaying_energy_level


    def _add_decayed_energy_level(self, decayed_energy_level )-> None:
        '''
        Adds an energy level the instance can decay into.
        '''
        assert isinstance(decayed_energy_level, EnergyLevel)
        assert decayed_energy_level != self
        assert decayed_energy_level.associated_component == self.associated_component
        
        self.decayed_energy_levels[decayed_energy_level.name] = decayed_energy_level


    def _add_coupled_energy_level(self, coupled_energy_level ) -> None:
        
        assert isinstance(coupled_energy_level, EnergyLevel)
        assert coupled_energy_level != self
        assert coupled_energy_level.associated_component == self.associated_component
        
        self.coupled_energy_levels[coupled_energy_level.name] = coupled_energy_level



    # def delete_decayed_energy_level(self, decayed_energy_level) -> None:
    
    #     assert isinstance(decayed_energy_level, EnergyLevel)
    #     assert decayed_energy_level != self
    #     assert decayed_energy_level.associated_component == self.associated_component
        
    #     del self.decayed_energy_levels[decayed_energy_level.name] 
    # def delete_coupled_energy_level(self, coupled_energy_level) -> None:
        
    #     assert isinstance(coupled_energy_level, EnergyLevel)
    #     assert coupled_energy_level != self
    #     assert coupled_energy_level.associated_component == self.associated_component

    #     del self.coupled_energy_levels[coupled_energy_level.name] 


    # def delete_rabi_energy_level(self, rabi_energy_level) -> None:
    
    #     assert isinstance(rabi_energy_level, EnergyLevel)
    #     assert rabi_energy_level != self
    #     assert rabi_energy_level.associated_component == self.associated_component
        
    #     del self.rabi_energy_levels[rabi_energy_level.name] 