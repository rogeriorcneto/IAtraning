# Configuração de Cores Corporativas SYNAPSE
# Baseado na logo da empresa SYNAPSE

class SynapseColors:
    """Paleta de cores oficial da SYNAPSE"""
    
    # Cores Principais da SYNAPSE
    SYNAPSE_BLUE_DARK = "#1f5582"      # Azul escuro principal (círculo da logo)
    SYNAPSE_BLUE_MEDIUM = "#2980b9"    # Azul médio (gradiente)
    SYNAPSE_BLUE_LIGHT = "#3498db"     # Azul claro (códigos binários)
    SYNAPSE_GRAY = "#5a6c7d"           # Cinza do texto SYNAPSE
    
    # Cores de Suporte
    SYNAPSE_WHITE = "#ffffff"
    SYNAPSE_LIGHT_GRAY = "#ecf0f1"
    SYNAPSE_DARK_GRAY = "#34495e"
    
    # Gradientes SYNAPSE
    GRADIENT_PRIMARY = f"linear-gradient(135deg, {SYNAPSE_BLUE_DARK} 0%, {SYNAPSE_BLUE_MEDIUM} 50%, {SYNAPSE_BLUE_LIGHT} 100%)"
    GRADIENT_SECONDARY = f"linear-gradient(45deg, {SYNAPSE_BLUE_DARK}, {SYNAPSE_BLUE_MEDIUM})"
    GRADIENT_LIGHT = f"linear-gradient(135deg, {SYNAPSE_BLUE_MEDIUM}, {SYNAPSE_BLUE_LIGHT})"
    
    # Cores com Transparência
    SYNAPSE_BLUE_ALPHA_10 = f"rgba(31, 85, 130, 0.1)"
    SYNAPSE_BLUE_ALPHA_30 = f"rgba(31, 85, 130, 0.3)"
    SYNAPSE_BLUE_ALPHA_50 = f"rgba(31, 85, 130, 0.5)"
    
    # Cores para Estados
    SUCCESS = "#27ae60"    # Verde para sucessos
    WARNING = "#f39c12"    # Laranja para avisos  
    ERROR = "#e74c3c"      # Vermelho para erros
    INFO = SYNAPSE_BLUE_LIGHT  # Azul SYNAPSE para informações
    
    @classmethod
    def get_css_variables(cls):
        """Retorna variáveis CSS com as cores SYNAPSE"""
        return f"""
        :root {{
            --synapse-blue-dark: {cls.SYNAPSE_BLUE_DARK};
            --synapse-blue-medium: {cls.SYNAPSE_BLUE_MEDIUM};
            --synapse-blue-light: {cls.SYNAPSE_BLUE_LIGHT};
            --synapse-gray: {cls.SYNAPSE_GRAY};
            --synapse-white: {cls.SYNAPSE_WHITE};
            --synapse-gradient-primary: {cls.GRADIENT_PRIMARY};
            --synapse-gradient-secondary: {cls.GRADIENT_SECONDARY};
            --synapse-success: {cls.SUCCESS};
            --synapse-warning: {cls.WARNING};
            --synapse-error: {cls.ERROR};
            --synapse-info: {cls.INFO};
        }}
        """
    
    @classmethod
    def get_plotly_colors(cls):
        """Retorna cores para gráficos Plotly"""
        return [
            cls.SYNAPSE_BLUE_DARK,
            cls.SYNAPSE_BLUE_MEDIUM, 
            cls.SYNAPSE_BLUE_LIGHT,
            cls.SYNAPSE_GRAY,
            cls.SUCCESS,
            cls.WARNING,
            cls.ERROR
        ]
    
    @classmethod
    def get_matplotlib_colors(cls):
        """Retorna cores para gráficos Matplotlib"""
        return {
            'primary': cls.SYNAPSE_BLUE_DARK,
            'secondary': cls.SYNAPSE_BLUE_MEDIUM,
            'accent': cls.SYNAPSE_BLUE_LIGHT,
            'neutral': cls.SYNAPSE_GRAY,
            'success': cls.SUCCESS,
            'warning': cls.WARNING,
            'error': cls.ERROR
        }

# Função de conveniência para usar as cores
def get_synapse_color(color_name):
    """
    Função helper para obter cores SYNAPSE
    
    Args:
        color_name (str): Nome da cor ('primary', 'secondary', 'light', 'gray', etc.)
    
    Returns:
        str: Código hexadecimal da cor
    """
    color_map = {
        'primary': SynapseColors.SYNAPSE_BLUE_DARK,
        'secondary': SynapseColors.SYNAPSE_BLUE_MEDIUM,
        'light': SynapseColors.SYNAPSE_BLUE_LIGHT,
        'gray': SynapseColors.SYNAPSE_GRAY,
        'white': SynapseColors.SYNAPSE_WHITE,
        'success': SynapseColors.SUCCESS,
        'warning': SynapseColors.WARNING,
        'error': SynapseColors.ERROR,
        'info': SynapseColors.INFO
    }
    
    return color_map.get(color_name, SynapseColors.SYNAPSE_BLUE_DARK)

# Teste das cores (se executado diretamente)
if __name__ == "__main__":
    print("🎨 Paleta de Cores SYNAPSE")
    print("=" * 40)
    print(f"Azul Escuro:  {SynapseColors.SYNAPSE_BLUE_DARK}")
    print(f"Azul Médio:   {SynapseColors.SYNAPSE_BLUE_MEDIUM}")
    print(f"Azul Claro:   {SynapseColors.SYNAPSE_BLUE_LIGHT}")
    print(f"Cinza:        {SynapseColors.SYNAPSE_GRAY}")
    print(f"Sucesso:      {SynapseColors.SUCCESS}")
    print(f"Aviso:        {SynapseColors.WARNING}")
    print(f"Erro:         {SynapseColors.ERROR}")
    print("=" * 40)
    print("Cores carregadas com sucesso! ✅")
