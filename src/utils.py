
import ifcopenshell


# create a new blank ifc file
def setup_ifc_file(blueprint):
    ifc = ifcopenshell.open(blueprint)
    ifcNew = ifcopenshell.file(schema=ifc.schema)

    owner_history = ifc.by_type("IfcOwnerHistory")[0]
    project = ifc.by_type("IfcProject")[0]
    context = ifc.by_type("IfcGeometricRepresentationContext")[0]
    floor = ifc.by_type("IfcBuildingStorey")[0]

    ifcNew.add(project)
    ifcNew.add(owner_history)
    ifcNew.add(context)
    ifcNew.add(floor)

    return ifcNew
