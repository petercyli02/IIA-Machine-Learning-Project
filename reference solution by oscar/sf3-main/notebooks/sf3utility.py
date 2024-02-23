import matplotlib.pyplot as plt


def setup_phase_portrait( ax ):

    # show the grid
    ax.grid( visible = True, zorder=1 )
    
    ax.axhline( color="black", zorder=2 )
    ax.axvline( color="black", zorder=2 )

