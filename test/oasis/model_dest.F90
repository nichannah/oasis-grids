
program model_dest

use coupler, only : coupler_init, coupler_init_done, coupler_close, &
                    coupler_get, coupler_put, coupler_add_field, &
                    coupler_destroy_field, couple_field_type, COUPLER_IN

implicit none

    type(couple_field_type), allocatable(:) :: fields

    ! Initialise the coupler.
    call coupler_init('destxx', 360, 300, 1, 1)

    allocate(fields(1))

    ! Create/add the coupling field.
    call coupler_add_field(field(1), 'field', COUPLER_IN)
    call coupler_init_done()

    ! Get fields from coupler and write out.
    call coupler_get(curr_time, fields)

    call coupler_destroy_field(fields(1))

    deallocate(fields)

    call coupler_close()

end program
