
program model_src

use coupler, only : coupler_init, coupler_init_done, coupler_add_field, &
                    coupler_destroy_field, coupler_put, coupler_close, &
                    coupler_destroy_field, couple_field_type, &
                    COUPLER_OUT, coupler_dump_field
implicit none

    type(couple_field_type), dimension(:), allocatable :: fields
    integer :: i, j, counter

    ! Initialise the coupler.
    call coupler_init('srcxxx', 192, 94, 1, 1)

    allocate(fields(1))

    ! Create/add the coupling field.
    call coupler_add_field(fields(1), 'src_field', COUPLER_OUT)
    call coupler_init_done()

    ! Initialise the field.
    counter = 0
    do j=1, size(fields(1)%field, 2)
      do i=1, size(fields(1)%field, 1)
        fields(1)%field(i, j) = counter
        counter = counter + 1
      enddo
    enddo

    call coupler_dump_field(fields(1), 'src_field.nc')

    call coupler_put(0, fields)

    call coupler_destroy_field(fields(1))
    call coupler_close()

end program model_src
