# Item_Identifier
# Item_Weight
# Item_Fat_Content
# Item_Visibility
# Item_Type
# Item_MRP
# Outlet_Identifier
# Outlet_Establishment_Year
# Outlet_Size
# Outlet_Location_Type
# Outlet_Type

import subprocess
from prompt_toolkit import prompt
from prompt_toolkit.contrib.completers import WordCompleter
from prompt_toolkit.styles import style_from_dict
from prompt_toolkit.token import Token
from prompt_toolkit.validation import Validator, ValidationError
from prompt_toolkit.shortcuts import print_tokens

features = ['Item_Identifier',
            'Item_Weight',
            'Item_Fat_Content',
            'Item_Visibility',
            'Item_Type',
            'Item_MRP',
            'Outlet_Identifier',
            'Outlet_Establishment_Year',
            'Outlet_Size',
            'Outlet_Location_Type',
            'Outlet_Type',
            'Item_Outlet_Sales'
            ]

dtypes = {
    "Item_Identifier": "string",
    "Item_Weight": "float",
    "Item_Fat_Content": "string",
    "Item_Visibility": "float",
    "Item_Type": "string",
    "Item_MRP": "float",
    "Outlet_Identifier": "string",
    "Outlet_Establishment_Year": "int",
    "Outlet_Size": "string",
    "Outlet_Location_Type": "string",
    "Outlet_Type": "string",
    "Item_Outlet_Sales": "float",
}

category = dict()

category["Item_Identifier"] = ['NCW54', 'NCQ41', 'NCS42', 'FDN45', 'FDX19',
                               'FDW47', 'NCA17', 'NCM05', 'FDK56', 'FDP09',
                               'FDO22', 'FDD46', 'NCD06', 'FDL13', 'FDR47',
                               'FDV23', 'FDU23', 'FDK03', 'FDR45', 'FDA44',
                               'FDR44', 'FDX27', 'NCA30', 'FDO23', 'FDH33',
                               'FDV31', 'DRF25', 'NCM55', 'FDG08', 'NCV17',
                               'FDY11', 'FDS09', 'FDU34', 'NCN42', 'FDH47',
                               'FDU46', 'NCC30', 'FDQ33', 'FDV24', 'NCE55',
                               'FDW07', 'FDR20', 'DRF27', 'DRI11', 'FDZ14',
                               'FDR49', 'FDX32', 'DRE15', 'NCK54', 'FDQ49',
                               'FDO03', 'FDZ51', 'FDP51', 'FDU49', 'FDS49',
                               'DRI47', 'FDG57', 'NCU53', 'DRE37', 'NCS29',
                               'FDY16', 'NCY41', 'FDM16', 'FDA23', 'NCM30',
                               'FDY58', 'NCR05', 'FDD11', 'FDX10', 'FDG32',
                               'FDC22', 'FDG17', 'FDY28', 'NCQ06', 'FDS43',
                               'FDT34', 'NCI54', 'FDU35', 'FDV52', 'NCU06',
                               'FDQ51', 'FDX15', 'FDO01', 'NCO43', 'FDR27',
                               'NCR29', 'FDV07', 'FDW38', 'FDL04', 'FDR60',
                               'FDC44', 'FDI27', 'FDG59', 'NCT53', 'FDB44',
                               'NCA29', 'FDR43', 'FDQ25', 'NCB06', 'NCW18',
                               'DRE49', 'FDE58', 'FDC47', 'NCI29', 'FDZ49',
                               'NCN07', 'FDM40', 'NCX29', 'FDZ38', 'FDJ45',
                               'FDW13', 'FDW15', 'FDC38', 'NCV05', 'FDE38',
                               'FDV59', 'FDG56', 'FDK46', 'FDR40', 'FDS08',
                               'FDS23', 'FDU45', 'FDO34', 'FDL21', 'FDK48',
                               'FDP38', 'NCO02', 'DRB13', 'DRI59', 'FDS58',
                               'FDV02', 'FDC16', 'FDA01', 'FDA34', 'FDZ60',
                               'FDS04', 'FDZ28', 'FDM02', 'NCP43', 'FDT26',
                               'FDO48', 'FDA38', 'FDM58', 'NCJ29', 'FDX31',
                               'FDF26', 'FDK36', 'FDF08', 'FDE40', 'NCU42',
                               'NCG42', 'FDW45', 'NCY05', 'FDQ45', 'FDR23',
                               'FDT46', 'FDL02', 'FDW58', 'FDB05', 'FDI32',
                               'DRB24', 'FDI52', 'NCM42', 'DRJ51', 'DRC25',
                               'FDA35', 'DRD27', 'FDU47', 'FDA31', 'FDY12',
                               'FDI12', 'FDM57', 'NCC43', 'NCA41', 'FDS35',
                               'NCT41', 'NCR38', 'NCE07', 'FDV50', 'DRI23',
                               'FDD08', 'FDU26', 'FDG05', 'FDH52', 'FDA47',
                               'FDL33', 'FDX24', 'FDP01', 'FDZ25', 'DRJ49',
                               'NCN54', 'NCQ43', 'FDV03', 'FDI48', 'FDD52',
                               'FDV33', 'FDS01', 'FDZ19', 'FDJ14', 'FDG28',
                               'FDD21', 'FDL14', 'NCI17', 'FDA25', 'FDJ04',
                               'FDU57', 'FDS15', 'FDW50', 'FDU20', 'FDE26',
                               'FDV57', 'FDU43', 'FDM39', 'NCL30', 'FDR26',
                               'FDK09', 'FDG33', 'FDT60', 'FDD56', 'FDR25',
                               'NCL06', 'NCH55', 'FDG26', 'FDS56', 'FDZ15',
                               'FDV26', 'FDP36', 'FDN46', 'NCL05', 'FDE35',
                               'NCF55', 'FDN52', 'FDC11', 'FDY37', 'FDK02',
                               'DRE01', 'FDX50', 'FDH50', 'FDQ26', 'FDG22',
                               'FDL27', 'FDB28', 'NCG30', 'FDR59', 'NCH43',
                               'FDT48', 'FDL25', 'DRH51', 'FDH10', 'FDO11',
                               'NCH42', 'FDI34', 'FDG34', 'FDE16', 'DRL23',
                               'DRC13', 'NCO42', 'FDU48', 'FDJ32', 'NCD31',
                               'FDX55', 'FDE23', 'FDB39', 'FDK20', 'NCB42',
                               'FDF40', 'NCJ30', 'DRF36', 'NCQ53', 'FDE46',
                               'NCU17', 'NCS06', 'FDR48', 'FDD35', 'FDO24',
                               'FDD04', 'FDW46', 'FDY33', 'NCZ17', 'FDX40',
                               'NCZ53', 'FDN48', 'DRH25', 'FDP04', 'FDX13',
                               'NCD54', 'FDV27', 'FDT50', 'FDF41', 'FDU50',
                               'FDQ15', 'FDD39', 'FDV51', 'FDT04', 'NCQ54',
                               'FDT59', 'NCA42', 'FDQ40', 'NCW53', 'FDJ36',
                               'FDI22', 'FDP37', 'FDW40', 'FDO49', 'NCO53',
                               'FDQ58', 'FDO60', 'FDN51', 'FDK22', 'DRP47',
                               'NCP50', 'NCL54', 'FDJ26', 'FDA16', 'FDM13',
                               'FDY46', 'NCE42', 'DRJ37', 'FDJ28', 'FDI50',
                               'FDH19', 'DRK59', 'FDV14', 'FDC09', 'FDT52',
                               'FDG16', 'FDO25', 'DRH37', 'NCB54', 'FDN20',
                               'FDQ39', 'FDM22', 'FDZ36', 'DRG48', 'FDL03',
                               'NCK19', 'NCX41', 'FDS55', 'FDC39', 'DRF51',
                               'FDY27', 'FDI56', 'NCD43', 'FDH60', 'FDS22',
                               'FDB23', 'FDC21', 'FDO46', 'NCS38', 'FDR19',
                               'FDS27', 'FDS16', 'FDJ08', 'NCX17', 'FDC17',
                               'DRJ23', 'FDQ20', 'FDS24', 'FDW20', 'FDC57',
                               'FDA15', 'NCN29', 'FDI60', 'DRD13', 'FDW10',
                               'FDZ07', 'FDX02', 'FDG10', 'FDO09', 'NCV18',
                               'FDX23', 'FDT51', 'DRN11', 'NCG54', 'FDJ60',
                               'FDT33', 'FDB16', 'FDK25', 'FDT09', 'FDV55',
                               'FDO04', 'FDW31', 'FDZ52', 'FDV36', 'FDD05',
                               'FDT58', 'FDM36', 'FDK45', 'FDQ10', 'FDB60',
                               'FDN23', 'NCN14', 'FDF02', 'FDX04', 'FDP20',
                               'FDN22', 'FDU19', 'FDF34', 'FDJ21', 'DRM48',
                               'NCK17', 'FDJ55', 'FDG21', 'FDI35', 'DRE03',
                               'DRF15', 'FDZ44', 'FDO38', 'DRJ01', 'FDX34',
                               'DRK23', 'FDA04', 'DRF01', 'NCT42', 'NCL29',
                               'FDY07', 'FDY38', 'NCO17', 'DRL35', 'FDD48',
                               'NCL55', 'FDS19', 'FDS20', 'FDE04', 'FDY36',
                               'FDI40', 'NCW41', 'FDC46', 'FDF10', 'NCQ05',
                               'FDP33', 'DRD01', 'FDP24', 'FDA40', 'NCN05',
                               'FDC14', 'FDF45', 'FDD40', 'FDW24', 'FDF47',
                               'FDQ57', 'FDJ34', 'FDF50', 'FDY57', 'FDZ37',
                               'DRM49', 'NCZ30', 'FDO27', 'FDE47', 'FDX45',
                               'FDZ55', 'FDT28', 'FDX11', 'FDZ10', 'NCI55',
                               'NCC06', 'NCD55', 'NCM19', 'FDZ46', 'FDE02',
                               'NCT06', 'FDC05', 'FDY24', 'FDG53', 'FDH58',
                               'FDQ48', 'NCP53', 'FDK08', 'NCE06', 'DRC49',
                               'FDA09', 'FDY49', 'NCH29', 'FDT57', 'FDQ34',
                               'FDN21', 'FDW22', 'FDZ03', 'FDZ27', 'NCA18',
                               'FDD57', 'FDJ33', 'FDY26', 'FDX56', 'FDG44',
                               'FDP11', 'FDC58', 'FDA57', 'NCO07', 'FDD20',
                               'FDL48', 'FDA10', 'NCY53', 'NCR50', 'DRM47',
                               'FDY47', 'FDV16', 'DRA24', 'FDP52', 'FDY51',
                               'DRH36', 'FDI41', 'FDQ24', 'FDN16', 'FDU21',
                               'DRG15', 'NCP18', 'NCU18', 'FDU14', 'NCN43',
                               'FDI44', 'FDP23', 'FDS13', 'FDF38', 'FDR07',
                               'FDZ31', 'FDX08', 'FDY08', 'NCN18', 'DRK11',
                               'FDG38', 'DRN35', 'FDB58', 'FDV21', 'DRH11',
                               'FDF28', 'FDM24', 'FDH04', 'NCY42', 'FDA13',
                               'NCJ31', 'DRH59', 'NCV29', 'FDE20', 'FDF04',
                               'FDY34', 'NCE43', 'DRN59', 'DRG36', 'FDS03',
                               'FDC23', 'FDN33', 'NCJ43', 'FDW08', 'NCR41',
                               'NCZ54', 'DRP35', 'FDR04', 'FDN58', 'FDS36',
                               'FDY32', 'FDG12', 'FDX37', 'FDR01', 'FDV43',
                               'FDX36', 'FDL43', 'FDW37', 'FDC51', 'FDQ14',
                               'FDR35', 'DRI13', 'FDW56', 'FDA37', 'DRH15',
                               'FDV11', 'FDB59', 'FDT27', 'FDY39', 'FDU09',
                               'FDZ04', 'FDN57', 'FDG46', 'FDO31', 'FDM03',
                               'FDK34', 'FDT38', 'FDH38', 'FDB10', 'FDU08',
                               'FDQ27', 'DRF13', 'NCW42', 'NCP02', 'NCI30',
                               'FDK10', 'FDO12', 'FDU36', 'FDG58', 'FDV37',
                               'DRA12', 'FDS45', 'FDF57', 'FDB29', 'NCM31',
                               'FDD53', 'FDI07', 'FDU38', 'FDJ41', 'NCV54',
                               'FDR55', 'FDH56', 'FDB51', 'NCS53', 'NCH06',
                               'DRH49', 'FDE52', 'FDV09', 'DRG39', 'FDE08',
                               'FDQ47', 'FDC08', 'FDJ38', 'FDF24', 'FDW34',
                               'FDD44', 'NCX54', 'FDE34', 'FDY14', 'NCP06',
                               'FDK14', 'FDM12', 'FDE45', 'NCB19', 'FDC04',
                               'FDS44', 'FDB14', 'FDS26', 'FDU15', 'FDN08',
                               'DRE13', 'NCL19', 'FDB49', 'FDM04', 'FDZ45',
                               'NCZ41', 'FDR16', 'FDZ21', 'FDG31', 'FDX60',
                               'NCV53', 'FDA45', 'FDY60', 'FDJ57', 'FDT45',
                               'FDY15', 'FDL57', 'FDV25', 'NCR30', 'NCS18',
                               'FDH24', 'FDT08', 'FDI20', 'NCP54', 'NCT17',
                               'FDO08', 'FDP44', 'FDT23', 'NCG55', 'FDV12',
                               'FDK60', 'NCL53', 'FDE36', 'FDC53', 'FDF56',
                               'NCP55', 'FDQ55', 'FDB38', 'FDZ13', 'FDX09',
                               'FDJ09', 'FDC50', 'FDP26', 'FDN10', 'FDC10',
                               'FDW27', 'FDT32', 'FDX44', 'FDZ12', 'FDX52',
                               'FDG41', 'FDS33', 'FDL50', 'FDM27', 'FDY50',
                               'NCI18', 'FDE05', 'FDE28', 'NCJ42', 'NCT54',
                               'NCD30', 'FDH09', 'FDL28', 'FDT14', 'FDT16',
                               'FDV32', 'FDM38', 'FDT44', 'FDA28', 'DRE48',
                               'FDG24', 'FDA58', 'FDH17', 'NCW06', 'NCD07',
                               'FDC37', 'FDY02', 'FDL34', 'FDP39', 'FDN12',
                               'FDI08', 'FDZ09', 'NCW30', 'NCS54', 'FDX51',
                               'FDP19', 'FDN24', 'FDR33', 'NCF42', 'FDZ26',
                               'FDA21', 'FDP22', 'DRI01', 'FDM08', 'FDQ59',
                               'DRZ24', 'FDT22', 'FDV40', 'FDN31', 'NCL41',
                               'FDD36', 'NCN55', 'NCF19', 'NCB31', 'NCE19',
                               'FDB52', 'FDY20', 'FDF39', 'FDE14', 'FDT25',
                               'FDF46', 'FDQ31', 'FDT21', 'FDQ16', 'FDJ22',
                               'FDG50', 'NCL07', 'DRN47', 'DRH13', 'FDN38',
                               'NCY54', 'FDQ22', 'NCQ02', 'FDS39', 'FDD38',
                               'NCX53', 'FDV56', 'FDU40', 'DRO47', 'NCX18',
                               'FDO15', 'FDI10', 'FDT37', 'FDH45', 'DRM11',
                               'FDA36', 'FDS57', 'NCS30', 'FDS37', 'FDO13',
                               'NCJ18', 'FDX48', 'FDR02', 'NCH54', 'FDG14',
                               'FDM51', 'FDU12', 'NCK31', 'FDN50', 'FDG45',
                               'FDH21', 'NCG43', 'FDO28', 'NCO26', 'FDU27',
                               'FDX25', 'NCS17', 'FDC41', 'FDU33', 'FDP16',
                               'FDL36', 'FDR12', 'DRL59', 'DRD37', 'FDJ56',
                               'DRK47', 'FDR52', 'NCF07', 'FDX33', 'FDI36',
                               'DRG13', 'FDY48', 'FDQ21', 'NCN30', 'FDK51',
                               'NCA05', 'FDM45', 'FDF11', 'FDO10', 'FDZ57',
                               'FDZ16', 'FDW26', 'NCQ50', 'FDR08', 'FDT43',
                               'FDV01', 'FDK44', 'FDS10', 'FDD14', 'FDR36',
                               'FDW33', 'FDH16', 'FDP12', 'FDV44', 'FDP25',
                               'FDT36', 'FDJ07', 'FDT24', 'FDU13', 'FDF20',
                               'FDI57', 'FDD03', 'FDS28', 'FDS47', 'FDE11',
                               'FDO51', 'NCO06', 'FDO57', 'NCF18', 'DRH03',
                               'DRI49', 'NCI31', 'FDM01', 'FDW16', 'DRC27',
                               'FDA14', 'FDA48', 'DRQ35', 'FDU03', 'FDM34',
                               'FDO37', 'FDI38', 'FDA19', 'FDM50', 'NCZ29',
                               'FDY56', 'NCN19', 'NCN26', 'NCN53', 'FDK41',
                               'FDU37', 'FDW44', 'FDY19', 'FDC20', 'FDV35',
                               'FDD50', 'FDT19', 'FDK21', 'FDI05', 'FDC40',
                               'FDA50', 'FDW35', 'FDS12', 'FDQ44', 'FDC33',
                               'NCA53', 'FDJ12', 'FDI33', 'FDN09', 'NCV41',
                               'NCJ17', 'FDV19', 'NCM54', 'NCR17', 'FDX22',
                               'DRY23', 'FDF35', 'FDV22', 'FDK32', 'FDX20',
                               'FDS11', 'FDV58', 'FDT12', 'NCQ42', 'FDT31',
                               'FDA20', 'FDD23', 'NCK18', 'FDO40', 'FDB46',
                               'FDO19', 'FDW19', 'FDG35', 'FDM52', 'DRH39',
                               'FDB20', 'FDC60', 'FDE44', 'FDQ11', 'FDR22',
                               'FDE57', 'NCG19', 'FDU31', 'FDI21', 'NCR42',
                               'FDQ32', 'FDN56', 'FDR32', 'FDW01', 'NCH18',
                               'FDP15', 'NCX30', 'NCT18', 'NCY17', 'FDW11',
                               'FDV49', 'FDS48', 'FDC56', 'DRC12', 'DRF37',
                               'FDE56', 'NCY29', 'FDF52', 'FDV08', 'DRC36',
                               'DRI37', 'FDS59', 'FDB56', 'FDD16', 'NCO14',
                               'FDL56', 'FDI04', 'NCC55', 'FDW59', 'FDF21',
                               'NCC19', 'FDT47', 'FDZ02', 'FDF22', 'NCK30',
                               'FDF16', 'FDP48', 'FDJ58', 'FDV45', 'NCO54',
                               'FDK26', 'FDN49', 'FDU25', 'NCU54', 'FDH22',
                               'FDC28', 'FDK16', 'FDK33', 'FDP21', 'FDK24',
                               'FDS07', 'FDX58', 'NCC31', 'FDJ53', 'FDA11',
                               'FDV46', 'NCQ18', 'FDU51', 'FDA49', 'FDH32',
                               'DRM37', 'DRE60', 'NCF06', 'FDL46', 'FDW28',
                               'FDU10', 'FDD29', 'DRK13', 'NCV42', 'NCE18',
                               'FDA26', 'NCX42', 'FDB47', 'FDN60', 'FDA52',
                               'FDQ28', 'FDB09', 'FDS02', 'FDG52', 'FDY03',
                               'FDB32', 'FDE22', 'FDK43', 'NCN06', 'NCC42',
                               'NCM06', 'FDH14', 'FDJ48', 'FDZ59', 'FDU28',
                               'FDB34', 'FDQ03', 'FDO33', 'FDY45', 'FDG02',
                               'FDQ01', 'FDE41', 'FDF17', 'NCI43', 'FDC32',
                               'FDL58', 'NCQ38', 'FDI28', 'DRJ24', 'FDZ58',
                               'DRF60', 'FDJ27', 'FDB15', 'NCP41', 'FDU07',
                               'NCP05', 'NCG07', 'FDW09', 'FDC34', 'DRJ39',
                               'FDI14', 'NCM41', 'FDZ40', 'FDH31', 'FDZ35',
                               'NCR53', 'FDP31', 'NCW17', 'FDJ46', 'NCA06',
                               'FDX14', 'DRE27', 'DRJ35', 'NCM43', 'DRO59',
                               'FDL24', 'FDU11', 'FDV28', 'FDL52', 'FDB21',
                               'FDW25', 'FDF12', 'FDB37', 'FDW55', 'FDZ23',
                               'FDD17', 'FDR57', 'FDS32', 'FDQ60', 'FDL15',
                               'DRK37', 'FDQ07', 'FDP57', 'FDM21', 'NCM18',
                               'NCN17', 'NCX05', 'FDC26', 'FDH20', 'FDE24',
                               'NCZ18', 'DRJ25', 'FDR03', 'FDT01', 'FDE17',
                               'FDU16', 'FDU60', 'FDD41', 'FDZ22', 'FDY35',
                               'FDU52', 'FDH34', 'FDU56', 'FDL10', 'FDD26',
                               'DRG11', 'FDT03', 'FDO16', 'FDL09', 'FDX01',
                               'FDL44', 'FDU01', 'FDM14', 'FDH44', 'DRL47',
                               'NCK42', 'NCO55', 'FDF05', 'FDW14', 'NCM53',
                               'FDM09', 'FDB03', 'FDB02', 'FDL40', 'FDW03',
                               'DRE12', 'FDB17', 'FDA03', 'FDP03', 'FDK38',
                               'FDM10', 'FDC59', 'FDU04', 'NCN41', 'DRC24',
                               'FDK50', 'FDG40', 'DRD25', 'NCP14', 'FDE39',
                               'DRD15', 'FDT56', 'FDW57', 'FDC45', 'FDQ09',
                               'FDN01', 'FDC52', 'DRG23', 'DRD49', 'FDU22',
                               'FDO56', 'FDA46', 'FDX39', 'FDB04', 'FDZ48',
                               'NCU30', 'NCF54', 'FDP28', 'FDU32', 'FDD10',
                               'FDQ19', 'FDX35', 'FDP07', 'DRJ59', 'NCY30',
                               'FDM46', 'FDP32', 'DRF49', 'FDX28', 'FDH46',
                               'FDX03', 'FDR11', 'FDB57', 'FDL38', 'FDA27',
                               'FDV34', 'FDO32', 'FDU39', 'FDG09', 'FDN13',
                               'FDL32', 'FDL45', 'FDB41', 'FDP27', 'FDH35',
                               'FDW36', 'FDD59', 'NCO41', 'FDB40', 'FDM32',
                               'FDE53', 'FDP10', 'DRI25', 'FDY10', 'FDY21',
                               'FDX49', 'FDA39', 'FDD09', 'FDA22', 'FDS31',
                               'NCG06', 'FDX46', 'NCR18', 'FDR51', 'DRL11',
                               'NCK53', 'FDW04', 'DRG03', 'FDU55', 'FDH57',
                               'FDL39', 'FDQ12', 'FDR28', 'FDR46', 'FDX21',
                               'FDB33', 'FDI24', 'FDD34', 'FDZ43', 'NCI06',
                               'FDC03', 'FDA33', 'FDR34', 'FDM44', 'FDR56',
                               'FDL20', 'FDD47', 'NCP30', 'DRG27', 'FDP58',
                               'FDY59', 'FDV48', 'NCA54', 'FDP45', 'FDU44',
                               'FDV10', 'FDG60', 'DRB25', 'NCD18', 'FDJ16',
                               'FDY55', 'DRG51', 'FDZ56', 'FDV47', 'FDW48',
                               'FDM28', 'DRK01', 'NCQ30', 'DRI51', 'FDK04',
                               'NCQ29', 'NCG18', 'DRL49', 'DRJ13', 'FDU02',
                               'FDZ01', 'FDK40', 'FDT13', 'DRA59', 'FDV39',
                               'FDJ10', 'FDR24', 'FDD28', 'NCE54', 'FDA55',
                               'FDM56', 'NCR54', 'NCM07', 'FDH40', 'NCM29',
                               'FDP46', 'FDC29', 'FDY13', 'FDT39', 'FDN32',
                               'FDV13', 'FDJ44', 'NCH30', 'FDH53', 'FDY25',
                               'DRG01', 'FDU59', 'FDS60', 'FDY09', 'DRJ47',
                               'FDH02', 'FDO45', 'DRD12', 'FDN02', 'FDM20',
                               'FDM33', 'FDZ47', 'FDF59', 'FDT10', 'FDK57',
                               'NCB43', 'FDE10', 'FDI19', 'FDH27', 'DRO35',
                               'FDW02', 'FDO58', 'FDZ08', 'FDZ20', 'NCE30',
                               'NCV06', 'FDA56', 'NCE31', 'DRM35', 'FDI26',
                               'FDE50', 'FDP59', 'DRL37', 'FDT07', 'FDP13',
                               'DRI03', 'FDA08', 'FDR37', 'FDX16', 'FDO36',
                               'FDA32', 'FDF53', 'NCJ06', 'NCL17', 'FDV20',
                               'FDF09', 'FDE51', 'NCI42', 'FDE59', 'DRM59',
                               'DRH23', 'DRD24', 'DRE25', 'FDY01', 'NCY18',
                               'FDQ23', 'DRZ11', 'FDS50', 'NCS05', 'FDR39',
                               'FDK55', 'FDN27', 'FDD33', 'DRB01', 'FDC35',
                               'FDR10', 'NCU05', 'NCD19', 'FDR09', 'NCO30',
                               'FDV38', 'NCJ54', 'FDG04', 'DRN37', 'FDE33',
                               'FDQ08', 'FDT49', 'FDN40', 'FDT20', 'FDL26',
                               'NCT30', 'FDI45', 'FDJ20', 'DRG49', 'FDW12',
                               'FDQ52', 'FDQ46', 'NCF43', 'FDK58', 'NCT29',
                               'FDD02', 'FDP49', 'FDZ50', 'NCK06', 'FDP56',
                               'NCB55', 'FDE32', 'FDD32', 'NCZ06', 'FDQ13',
                               'DRF03', 'FDN34', 'FDW32', 'NCC18', 'NCD42',
                               'NCZ05', 'FDI58', 'NCO05', 'FDL22', 'NCL42',
                               'NCC07', 'DRK49', 'NCS41', 'NCP17', 'NCT05',
                               'FDI02', 'NCW05', 'FDZ32', 'FDF32', 'DRG37',
                               'FDV04', 'FDB22', 'FDT02', 'FDE09', 'FDH12',
                               'FDY22', 'FDY43', 'FDW51', 'FDX26', 'FDB12',
                               'FDS40', 'FDJ52', 'DRJ11', 'FDG47', 'FDK27',
                               'DRI39', 'FDZ33', 'FDA07', 'FDR15', 'FDT15',
                               'FDV15', 'FDX07', 'FDQ04', 'NCM17', 'FDP34',
                               'NCK29', 'FDW23', 'FDQ37', 'NCB30', 'FDB36',
                               'FDY44', 'FDJ40', 'FDE29', 'FDN28', 'FDJ03',
                               'FDO21', 'FDH05', 'FDK52', 'FDX47', 'FDF58',
                               'FDN25', 'FDO44', 'NCF30', 'FDM25', 'NCQ17',
                               'NCO29', 'DRB48', 'FDX38', 'FDG29', 'FDN39',
                               'FDO50', 'NCH07', 'DRM23', 'FDT55', 'NCR06',
                               'DRL60', 'FDY52', 'FDM15', 'FDX43', 'FDC15',
                               'NCO18', 'NCK07', 'FDD22', 'DRF23', 'FDY31',
                               'DRL01', 'FDY40', 'NCB18', 'FDF44', 'FDF33',
                               'FDL16', 'FDX59', 'FDA43', 'FDS21', 'FDB26',
                               'FDN15', 'FDR13', 'FDW39', 'FDH28', 'FDB45',
                               'FDS51', 'FDS46', 'FDP40', 'FDA51', 'FDB35',
                               'FDM60', 'FDC02', 'NCU41', 'FDN44', 'FDK28',
                               'FDS25', 'FDS34', 'FDJ02', 'FDI15', 'DRF48',
                               'NCJ19', 'FDN04', 'NCM26', 'FDI53', 'FDV60',
                               'FDW43', 'FDB53', 'DRG25', 'FDU58', 'DRK35',
                               'NCX06', 'FDQ56', 'FDB27', 'FDR21', 'FDR58',
                               'NCP42', 'FDZ39', 'NCP29', 'NCU29', 'FDD58',
                               'FDH08', 'FDH26', 'FDW52', 'FDP08', 'NCF31',
                               'NCC54', 'NCK05', 'FDI46', 'NCY06', 'NCL18',
                               'FDG20', 'FDA02', 'FDD51', 'FDT40', 'NCL31',
                               'FDO39', 'FDS14', 'NCW29', 'DRK12', 'FDR31',
                               'FDF14', 'FDO52', 'FDO20', 'FDB50', 'FDX57',
                               'FDH48', 'FDK15', 'NCB07', 'DRD60', 'FDF29',
                               'FDU24', 'FDN03', 'FDL12', 'FDP60', 'FDI16',
                               'FDE21', 'DRN36', 'DRK39', 'FDQ36', 'DRH01',
                               'FDJ50', 'FDS52', 'NCJ05', 'FDL08', 'FDI09',
                               'FDT35', 'FDJ15', 'FDB11', 'NCZ42', 'FDZ34',
                               'DRC01', 'NCV30', 'FDD45', 'FDW21', 'FDC48',
                               'FDL51', 'FDY04', 'FDB08', 'FDT11', 'FDW49',
                               'FDX12', 'FDH41', 'FDR14', 'FDW60']

category["Item_Weight"] = {}
category["Item_Weight"]["max"] = 21.35
category["Item_Weight"]["min"] = 4.555

category["Item_Fat_Content"] = ['Low Fat', 'reg', 'low fat', 'Regular', 'LF']

category["Item_Visibility"] = {}
category["Item_Visibility"]["max"] = 0.328390948
category["Item_Visibility"]["min"] = 0.0

category["Item_Type"] = ['Fruits and Vegetables', 'Baking Goods',
                         'Frozen Foods', 'Breads', 'Breakfast',
                         'Starchy Foods', 'Hard Drinks', 'Health and Hygiene',
                         'Household', 'Soft Drinks', 'Others', 'Canned',
                         'Meat', 'Snack Foods', 'Dairy', 'Seafood']

category["Item_MRP"] = {}
category["Item_MRP"]["max"] = 266.8884
category["Item_MRP"]["min"] = 31.29

category["Outlet_Identifier"] = ['OUT017', 'OUT013', 'OUT046', 'OUT019',
                                 'OUT027', 'OUT018', 'OUT045', 'OUT035',
                                 'OUT010', 'OUT049']

category["Outlet_Establishment_Year"] = {}
category["Outlet_Establishment_Year"]["max"] = 2009
category["Outlet_Establishment_Year"]["min"] = 1985

category["Outlet_Size"] = ['Medium', 'High', 'Small']

category["Outlet_Location_Type"] = ['Tier 1', 'Tier 2', 'Tier 3']

category["Outlet_Type"] = ['Supermarket Type1', 'Supermarket Type3',
                           'Supermarket Type2', 'Grocery Store']

category["Item_Outlet_Sales"] = {}
category["Item_Outlet_Sales"]["max"] = 13086.9648
category["Item_Outlet_Sales"]["min"] = 33.29

query = {}


class StringValidator(Validator):
    def validate(self, document):
        text = document.text

        if not text:
            raise ValidationError(
                message='Please fill the value. Press TAB for completion',
                cursor_position=len(text))


class NumberValidator(Validator):
    def validate(self, document):
        text = document.text

        if text and not text.isdigit():
            i = 0

            # Get index of fist non numeric character.
            # We want to move the cursor here.
            for i, c in enumerate(text):
                if not c.isdigit():
                    break

            raise ValidationError(
                message='This input contains non-numeric characters',
                cursor_position=i)


class FloatValidator(Validator):
    def validate(self, document):
        text = document.text
        try:
            float(text)
            validity = True
        except ValueError:
            validity = False

        if not validity:
            i = 0

            for i, c in enumerate(text):
                has_one_dot = False
                if c == '.' and not has_one_dot:
                    has_one_dot = True
                elif not c.isdigit():
                    break

            raise ValidationError(
                message='This input contains non-numeric characters',
                cursor_position=i)


prompt_style = style_from_dict({
    Token.Toolbar: '#ffffff bg:#333333',
    Token.Feature: '#FFFFFF',
    Token.DType: '#D95040',
    Token.Colon: '#FFFFFF',
    Token.Default: '#828282',
})


def prompt_number(content, feature):
    dtype = dtypes[feature]
    ft = feature.replace("_", " ")

    validators = {
        'string': StringValidator,
        'float': FloatValidator,
        'int': NumberValidator,
    }
    parser = {
        'string': str,
        'float': float,
        'int': int,
    }

    dict_completer = None
    if dtype == 'string':
        dict_completer = WordCompleter(content)

    selected_validator = validators[dtype]
    selected_parser = parser[dtype]
    default_value = None
    if dtype == 'string':
        default_value = content[0]
    elif dtype == 'int' or dtype == 'float':
        default_value = (content["max"] - content["min"] / 2) + content["min"]
        default_value = str(int(default_value))

    def get_bottom_toolbar_tokens(cli):
        selected_toolbar_text = "Whoa?! Please restart"
        if dtype == 'string':
            selected_toolbar_text = 'Press TAB for completion.'
        elif dtype == 'int' or dtype == 'float':
            selected_toolbar_text = 'Our data said that maximum is {} and minimum is {}'.format(
                content["max"], content["min"])

        return [(Token.Toolbar,
                 selected_toolbar_text)]

    def get_prompt_tokens(cli):
        return [
            (Token.Feature, "{} ".format(ft)),
            (Token.DType, "| {}".format(dtype)),
            (Token.Colon, ': '),
            (Token.Default, "({}) ".format(default_value)),
        ]

    query[feature] = selected_parser(
        prompt(
            get_prompt_tokens=get_prompt_tokens,
            validator=selected_validator(),
            default=default_value,
            get_bottom_toolbar_tokens=get_bottom_toolbar_tokens,
            completer=dict_completer,
            mouse_support=True,
            style=prompt_style))


if __name__ == '__main__':
    # Create a stylesheet.
    style = style_from_dict({
        Token.Hello: '#ff0066',
        Token.World: '#44ff44 bold',
    })

    # Make a list of (Token, text) tuples.
    tokens = [
        (Token.Hello, 'Prediksi Keuntungan Toko '),
        (Token.World, 'BigMart'),
        (Token, '\n'),
    ]

    # Print the result.
    print("")
    print_tokens(tokens, style=style)
    print("\n")

    for feature in features[0:-1]:
        isi = category[feature]
        prompt_number(isi, feature)

    input_query = ["python", "runner_cart.py",
                   str(query['Item_Weight']),
                   str(query['Item_Visibility']),
                   str(query['Item_MRP']),
                   str(query['Outlet_Establishment_Year']),
                   ]
    tokens_done = [
        (Token.Hello, '\nHasil Prediksi '),
        (Token.World, "Item Outlet Sales: "),
        (Token, '\n'),
    ]
    print_tokens(tokens_done, style=style)


    try:
        subprocess.call(input_query)
        raise SystemExit()

    except KeyboardInterrupt:
        print("Quite")


