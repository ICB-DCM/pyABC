from lxml import etree


def xmlGenerator(self, para: dict = None):
    self.param = para
    # create the file structure
    root = etree.Element('MorpheusModel')
    root.set('version', '3')

    # Fill xml file with description
    description = etree.SubElement(root, "Description")
    title = etree.SubElement(description, "Title")
    details = etree.SubElement(description, "Details")
    title.text = self.param['Title']
    details.text = self.param['Details']

    # Fill xml file with Global info
    etree.SubElement(root, "Global")

    # Fill xml file with Space info
    space = etree.SubElement(root, "Space")
    lattice = etree.SubElement(space, "Lattice")
    etree.SubElement(lattice, "Size")
    neighborhood = etree.SubElement(lattice, "Neighborhood")
    etree.SubElement(neighborhood, "Order")
    spaceSymbol = etree.SubElement(space, "SpaceSymbol")
    spaceSymbol.set("symbol", self.param['SpaceSymbolSymbol'])

    # Fill xml file with Time info
    time = etree.SubElement(root, "Time")
    startTime = etree.SubElement(time, "StartTime")
    startTime.set("value", self.param['StartTimeValue'])

    stopTime = etree.SubElement(time, "StopTime")
    stopTime.set("value", self.param['StopTimeValue'])

    timeSymbol = etree.SubElement(time, "TimeSymbol")
    timeSymbol.set("symbol", self.param['TimeSymbolSymbol'])

    # Fill xml file with CellTypes info
    cellTypes = etree.SubElement(root, "CellTypes")
    cellType = etree.SubElement(cellTypes, "CellType")
    cellType.set("class", self.param['CellTypeClass'])
    cellType.set("name", self.param['CellTypeName'])

    etree.SubElement(cellType, "Property")
    system = etree.SubElement(cellType, "System")
    etree.SubElement(system, "Constant")
    diffEqn = etree.SubElement(system, "DiffEqn")
    etree.SubElement(diffEqn, "Expression")

    # Fill xml file with CellPopulation info
    cellPopulation = etree.SubElement(root, "CellPopulation")
    population = etree.SubElement(cellPopulation, "Population")
    population.set("size", self.param['PopulationSize'])
    population.set("type", self.param['PopulationType'])
    etree.SubElement(cellPopulation, "Population")

    # Fill xml file with Analysis info
    etree.SubElement(root, "Analysis")

    # create and write the new XML file with the results
    etree.tostring(root)
    tree = etree.ElementTree(root)
    tree.write(
        "model.xml",
        pretty_print=True,
        xml_declaration=True,
        encoding='UTF-8')
