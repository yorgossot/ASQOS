import sympy

class EnergyLevel():
    
    def __init__(self, name : str, energy ) -> None:
        assert isinstance(name,str)

        self.name = name
        self.energy = sympy.sympify(energy)

    def __repr__(self) -> str:
        return f"Energy Level {str(self.name)} with energy E={str(self.energy)}" 

    def set_energy(self, energy ):
        self.energy = sympy.sympify(energy)
    
    
    def associate_with_component(self, component) -> None:
        
        # Import here to avoid circular import issues
        from .component import Component
        assert isinstance(component, Component)
        
        if not hasattr(self, 'associated_component'):
            self.associated_component = component
        else:
            raise Exception(f"Cannot associate {self} with component {component} since it \
                is already associated with {self.associated_component}")
    



