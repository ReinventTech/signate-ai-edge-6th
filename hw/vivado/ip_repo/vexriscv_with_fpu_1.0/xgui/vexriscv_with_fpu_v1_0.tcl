# Definitional proc to organize widgets for parameters.
proc init_gui { IPINST } {
  ipgui::add_param $IPINST -name "Component_Name"
  #Adding Page
  set Page_0 [ipgui::add_page $IPINST -name "Page 0"]
  ipgui::add_param $IPINST -name "VEXRISCV_RESET_ADDR" -parent ${Page_0}
  ipgui::add_param $IPINST -name "C_IMEM_AXI_TARGET_SLAVE_BASE_ADDR" -parent ${Page_0}
  ipgui::add_param $IPINST -name "C_IMEM_AXI_BURST_LEN" -parent ${Page_0} -widget comboBox
  ipgui::add_param $IPINST -name "C_IMEM_AXI_ID_WIDTH" -parent ${Page_0}
  ipgui::add_param $IPINST -name "C_IMEM_AXI_ADDR_WIDTH" -parent ${Page_0}
  ipgui::add_param $IPINST -name "C_IMEM_AXI_DATA_WIDTH" -parent ${Page_0} -widget comboBox
  ipgui::add_param $IPINST -name "C_DMEM_AXI_TARGET_SLAVE_BASE_ADDR" -parent ${Page_0}
  ipgui::add_param $IPINST -name "C_DMEM_AXI_BURST_LEN" -parent ${Page_0} -widget comboBox
  ipgui::add_param $IPINST -name "C_DMEM_AXI_ID_WIDTH" -parent ${Page_0}
  ipgui::add_param $IPINST -name "C_DMEM_AXI_ADDR_WIDTH" -parent ${Page_0}
  ipgui::add_param $IPINST -name "C_DMEM_AXI_DATA_WIDTH" -parent ${Page_0} -widget comboBox


}

proc update_PARAM_VALUE.VEXRISCV_RESET_ADDR { PARAM_VALUE.VEXRISCV_RESET_ADDR } {
	# Procedure called to update VEXRISCV_RESET_ADDR when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.VEXRISCV_RESET_ADDR { PARAM_VALUE.VEXRISCV_RESET_ADDR } {
	# Procedure called to validate VEXRISCV_RESET_ADDR
	return true
}

proc update_PARAM_VALUE.C_IMEM_AXI_TARGET_SLAVE_BASE_ADDR { PARAM_VALUE.C_IMEM_AXI_TARGET_SLAVE_BASE_ADDR } {
	# Procedure called to update C_IMEM_AXI_TARGET_SLAVE_BASE_ADDR when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.C_IMEM_AXI_TARGET_SLAVE_BASE_ADDR { PARAM_VALUE.C_IMEM_AXI_TARGET_SLAVE_BASE_ADDR } {
	# Procedure called to validate C_IMEM_AXI_TARGET_SLAVE_BASE_ADDR
	return true
}

proc update_PARAM_VALUE.C_IMEM_AXI_BURST_LEN { PARAM_VALUE.C_IMEM_AXI_BURST_LEN } {
	# Procedure called to update C_IMEM_AXI_BURST_LEN when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.C_IMEM_AXI_BURST_LEN { PARAM_VALUE.C_IMEM_AXI_BURST_LEN } {
	# Procedure called to validate C_IMEM_AXI_BURST_LEN
	return true
}

proc update_PARAM_VALUE.C_IMEM_AXI_ID_WIDTH { PARAM_VALUE.C_IMEM_AXI_ID_WIDTH } {
	# Procedure called to update C_IMEM_AXI_ID_WIDTH when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.C_IMEM_AXI_ID_WIDTH { PARAM_VALUE.C_IMEM_AXI_ID_WIDTH } {
	# Procedure called to validate C_IMEM_AXI_ID_WIDTH
	return true
}

proc update_PARAM_VALUE.C_IMEM_AXI_ADDR_WIDTH { PARAM_VALUE.C_IMEM_AXI_ADDR_WIDTH } {
	# Procedure called to update C_IMEM_AXI_ADDR_WIDTH when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.C_IMEM_AXI_ADDR_WIDTH { PARAM_VALUE.C_IMEM_AXI_ADDR_WIDTH } {
	# Procedure called to validate C_IMEM_AXI_ADDR_WIDTH
	return true
}

proc update_PARAM_VALUE.C_IMEM_AXI_DATA_WIDTH { PARAM_VALUE.C_IMEM_AXI_DATA_WIDTH } {
	# Procedure called to update C_IMEM_AXI_DATA_WIDTH when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.C_IMEM_AXI_DATA_WIDTH { PARAM_VALUE.C_IMEM_AXI_DATA_WIDTH } {
	# Procedure called to validate C_IMEM_AXI_DATA_WIDTH
	return true
}

proc update_PARAM_VALUE.C_DMEM_AXI_TARGET_SLAVE_BASE_ADDR { PARAM_VALUE.C_DMEM_AXI_TARGET_SLAVE_BASE_ADDR } {
	# Procedure called to update C_DMEM_AXI_TARGET_SLAVE_BASE_ADDR when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.C_DMEM_AXI_TARGET_SLAVE_BASE_ADDR { PARAM_VALUE.C_DMEM_AXI_TARGET_SLAVE_BASE_ADDR } {
	# Procedure called to validate C_DMEM_AXI_TARGET_SLAVE_BASE_ADDR
	return true
}

proc update_PARAM_VALUE.C_DMEM_AXI_BURST_LEN { PARAM_VALUE.C_DMEM_AXI_BURST_LEN } {
	# Procedure called to update C_DMEM_AXI_BURST_LEN when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.C_DMEM_AXI_BURST_LEN { PARAM_VALUE.C_DMEM_AXI_BURST_LEN } {
	# Procedure called to validate C_DMEM_AXI_BURST_LEN
	return true
}

proc update_PARAM_VALUE.C_DMEM_AXI_ID_WIDTH { PARAM_VALUE.C_DMEM_AXI_ID_WIDTH } {
	# Procedure called to update C_DMEM_AXI_ID_WIDTH when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.C_DMEM_AXI_ID_WIDTH { PARAM_VALUE.C_DMEM_AXI_ID_WIDTH } {
	# Procedure called to validate C_DMEM_AXI_ID_WIDTH
	return true
}

proc update_PARAM_VALUE.C_DMEM_AXI_ADDR_WIDTH { PARAM_VALUE.C_DMEM_AXI_ADDR_WIDTH } {
	# Procedure called to update C_DMEM_AXI_ADDR_WIDTH when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.C_DMEM_AXI_ADDR_WIDTH { PARAM_VALUE.C_DMEM_AXI_ADDR_WIDTH } {
	# Procedure called to validate C_DMEM_AXI_ADDR_WIDTH
	return true
}

proc update_PARAM_VALUE.C_DMEM_AXI_DATA_WIDTH { PARAM_VALUE.C_DMEM_AXI_DATA_WIDTH } {
	# Procedure called to update C_DMEM_AXI_DATA_WIDTH when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.C_DMEM_AXI_DATA_WIDTH { PARAM_VALUE.C_DMEM_AXI_DATA_WIDTH } {
	# Procedure called to validate C_DMEM_AXI_DATA_WIDTH
	return true
}


proc update_MODELPARAM_VALUE.C_IMEM_AXI_TARGET_SLAVE_BASE_ADDR { MODELPARAM_VALUE.C_IMEM_AXI_TARGET_SLAVE_BASE_ADDR PARAM_VALUE.C_IMEM_AXI_TARGET_SLAVE_BASE_ADDR } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.C_IMEM_AXI_TARGET_SLAVE_BASE_ADDR}] ${MODELPARAM_VALUE.C_IMEM_AXI_TARGET_SLAVE_BASE_ADDR}
}

proc update_MODELPARAM_VALUE.C_IMEM_AXI_BURST_LEN { MODELPARAM_VALUE.C_IMEM_AXI_BURST_LEN PARAM_VALUE.C_IMEM_AXI_BURST_LEN } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.C_IMEM_AXI_BURST_LEN}] ${MODELPARAM_VALUE.C_IMEM_AXI_BURST_LEN}
}

proc update_MODELPARAM_VALUE.C_IMEM_AXI_ID_WIDTH { MODELPARAM_VALUE.C_IMEM_AXI_ID_WIDTH PARAM_VALUE.C_IMEM_AXI_ID_WIDTH } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.C_IMEM_AXI_ID_WIDTH}] ${MODELPARAM_VALUE.C_IMEM_AXI_ID_WIDTH}
}

proc update_MODELPARAM_VALUE.C_IMEM_AXI_ADDR_WIDTH { MODELPARAM_VALUE.C_IMEM_AXI_ADDR_WIDTH PARAM_VALUE.C_IMEM_AXI_ADDR_WIDTH } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.C_IMEM_AXI_ADDR_WIDTH}] ${MODELPARAM_VALUE.C_IMEM_AXI_ADDR_WIDTH}
}

proc update_MODELPARAM_VALUE.C_IMEM_AXI_DATA_WIDTH { MODELPARAM_VALUE.C_IMEM_AXI_DATA_WIDTH PARAM_VALUE.C_IMEM_AXI_DATA_WIDTH } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.C_IMEM_AXI_DATA_WIDTH}] ${MODELPARAM_VALUE.C_IMEM_AXI_DATA_WIDTH}
}

proc update_MODELPARAM_VALUE.C_DMEM_AXI_TARGET_SLAVE_BASE_ADDR { MODELPARAM_VALUE.C_DMEM_AXI_TARGET_SLAVE_BASE_ADDR PARAM_VALUE.C_DMEM_AXI_TARGET_SLAVE_BASE_ADDR } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.C_DMEM_AXI_TARGET_SLAVE_BASE_ADDR}] ${MODELPARAM_VALUE.C_DMEM_AXI_TARGET_SLAVE_BASE_ADDR}
}

proc update_MODELPARAM_VALUE.C_DMEM_AXI_BURST_LEN { MODELPARAM_VALUE.C_DMEM_AXI_BURST_LEN PARAM_VALUE.C_DMEM_AXI_BURST_LEN } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.C_DMEM_AXI_BURST_LEN}] ${MODELPARAM_VALUE.C_DMEM_AXI_BURST_LEN}
}

proc update_MODELPARAM_VALUE.C_DMEM_AXI_ID_WIDTH { MODELPARAM_VALUE.C_DMEM_AXI_ID_WIDTH PARAM_VALUE.C_DMEM_AXI_ID_WIDTH } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.C_DMEM_AXI_ID_WIDTH}] ${MODELPARAM_VALUE.C_DMEM_AXI_ID_WIDTH}
}

proc update_MODELPARAM_VALUE.C_DMEM_AXI_ADDR_WIDTH { MODELPARAM_VALUE.C_DMEM_AXI_ADDR_WIDTH PARAM_VALUE.C_DMEM_AXI_ADDR_WIDTH } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.C_DMEM_AXI_ADDR_WIDTH}] ${MODELPARAM_VALUE.C_DMEM_AXI_ADDR_WIDTH}
}

proc update_MODELPARAM_VALUE.C_DMEM_AXI_DATA_WIDTH { MODELPARAM_VALUE.C_DMEM_AXI_DATA_WIDTH PARAM_VALUE.C_DMEM_AXI_DATA_WIDTH } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.C_DMEM_AXI_DATA_WIDTH}] ${MODELPARAM_VALUE.C_DMEM_AXI_DATA_WIDTH}
}

proc update_MODELPARAM_VALUE.VEXRISCV_RESET_ADDR { MODELPARAM_VALUE.VEXRISCV_RESET_ADDR PARAM_VALUE.VEXRISCV_RESET_ADDR } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.VEXRISCV_RESET_ADDR}] ${MODELPARAM_VALUE.VEXRISCV_RESET_ADDR}
}

