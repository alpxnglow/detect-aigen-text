import xml.etree.ElementTree as ET

# Parse the XML file
tree = ET.parse('data/0002b.xml')
root = tree.getroot()

# Find all div1 tags with type="section" and extract their content
div1_sections = root.findall('.//div1[@type="section"]')

print(div1_sections)

for div1_section in div1_sections:
    print("Extracting content from section:")
    for element in div1_section.iter():
        if element.tag != 'hi':
            tag = element.tag
            text = element.text.strip() if element.text else ''
            if text:
                print(f"Tag: {tag}, Text: {text}")



    # print("Extracting content from section:")
    # for elements in div1_section:
    #     for items in elements:
    #         if items.text and not any(child.tag == 'hi' for child in items):
    #             text = items.text.strip()
    #         try:
    #             print(f"{items.text}")
    #         except Exception as e:
    #             print(f"Encountered an exception: {e}")
    #             continue