import pyvista as pv
from pyvista.trame.ui import plotter_ui
from trame.app import get_server
from trame.ui.vuetify3 import SinglePageLayout
from trame.widgets import vuetify3 as vuetify

pv.OFF_SCREEN = True

server = get_server()
state, ctrl = server.state, server.controller

mesh = pv.read('data/vtk/u.vtu')

plotter = pv.Plotter(off_screen=True)
plotter.add_mesh(mesh, show_edges=True, cmap='viridis')
plotter.view_xy()

with SinglePageLayout(server) as layout:
    layout.icon.click = ctrl.view_reset_camera
    layout.title.set_text("JaxFEM Viewer")

    with layout.content:
        with vuetify.VContainer(
            fluid=True,
            classes="pa-0 fill-height",
        ):
            view = plotter_ui(plotter)
            ctrl.view_update = view.update

if __name__ == "__main__":
    server.start(port=8080)
